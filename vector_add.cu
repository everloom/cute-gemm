#include <cublas_v2.h>
#include <cuda.h>
#include <cute/tensor.hpp>
#include <float.h>
#include <stdlib.h>

/*
代码实现了z = ax + by + c的计算
使用了如下优化手段：
1、单个线程处理多个数据（一个线程处理8个数据），通过数据预取和指令并行，提升数据读取销量、提升执行单元的流水线效率；
2、对global内存进行大字长读写（使用LDG128，对应代码中的copy），减少数据IO所需要的指令数目，减少调度开销，提升程序运行效率；
3、使用Half2类型，减少half类型引入的PRMT指令的转换和开销；
4、使用FMA（fused multiply accumulate）指令完成计算，减少FMUL、FADD指令数，提升计算精度；
代码中一个线程处理8个数据
*/

/**
reed大佬博客中的原话https://zhuanlan.zhihu.com/p/663093816：
其中template行通过编译时常量指定每个线程处理8个数据，避免运行时常量不能利用寄存器（寄存器不可以寻址）所带来
的Local Memory问题。由于一个线程处理8个元素，并且单个数据的大小为sizeof(half) = 2, 这样
一个线程所需要的数据量为8 x 2 = 16byte，该大小的数据可以通过LDG.128指令实现一条指令完成对数据从全局内存到寄存器到加载；

这里主要解释一下“其中template行通过编译时常量指定每个线程处理8个数据，避免运行时常量不能利用寄存器（寄存器不可以寻址）所带来的Local Memory问题”这句话怎么理解
loal mem实际指的是gmem，首先来看如果kNumElemPerThread不是template中的编译时常量，而是运行时常量的问题
在代码中可以看到有下面这样的一个for循环，里面会按照i来访问txR2，这里txR2是kernel中定义的对象，长度大小就是kNumElemPerThread / 2
  Tensor tzr = local_tile(tz, make_shape(Int<kNumElemPerThread>{}), make_coord(idx));
  Tensor tzR = make_tensor_like(tzr);
  auto tzR2 = recast<half2>(tzR);  // 4个half2
  for (int i = 0; i < size(tzR2); ++i) {
    // two hfma2 instruction
    tzR2(i) = txR2(i) * a2 + (tyR2(i) * b2 + c2);
  }
可以看到这里txR2是通过index i来进行寻址访问的，但寄存器其实是不支持寻址访问的（寄存器不是连续的内存空间）
如果希望txR2能在寄存器上（是否在寄存器上由编译器决定），就需要让编译器提前知道txR2的长度，这样编译器
就可以对for循环进行循环展开来处理，得到类似下面的ptx代码
// 计算（完全展开，每个操作都指定具体寄存器）
hfma2.f16 %r16, %r0, %a2, ...;   // tzR2(0) = txR2(0) * a2 + ...
hfma2.f16 %r17, %r1, %a2, ...;   // tzR2(1) = txR2(1) * a2 + ...
hfma2.f16 %r18, %r2, %a2, ...;   // tzR2(2) = txR2(2) * a2 + ...
hfma2.f16 %r19, %r3, %a2, ...;   // tzR2(3) = txR2(3) * a2 + ...

但如果kNumElemPerThread是运行时常量，即kNumElemPerThread是kernel的一个传入参数，这样编译器在编译时
就不知道kNumElemPerThread的具体大小，也就不知道txR2的长度，这样编译器就没办法对for循环进行循环展开，
同时由于for循环中是寻址访问，local mem才支持寻址访问（gmem是连续的地址空间），所以编译器就会将txR2放到local mem中


// 循环被完全展开为：
tzR2(0) = ...;  // → 寄存器 r1
tzR2(1) = ...;  // → 寄存器 r2
tzR2(2) = ...;  // → 寄存器 r3
tzR2(3) = ...;  // → 寄存器 r4
// ... 每个元素都有独立寄存器
这样所有的数据都会在寄存器中，而不是移到local mem中去
*/
template <int kNumElemPerThread = 8>
__global__ void vector_add_local_tile_multi_elem_per_thread_half(
    half *z, int num, const half *x, const half *y, const half a, const half b, const half c) {
  using namespace cute;

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  // 这里num变量应该是vector的长度
  if (idx >= num / kNumElemPerThread) { // 未处理非对齐问题
    return;
  }

  // Tensor tz、tx、ty行，通过利用make_tensor 接口将kernel参数中的裸指针和维度信息包装成tensor表达
  // 这里的tz由于初始化时使用make tensor方法并传入了指针，所以是堆上对象(reed大佬博客里说这个是堆上对象)
  // 堆上对象的意思是，对象指向的数据放在堆上（gmem或者smem）
  // 然后问了下claude和豆包，它们说tz本身是栈上对象，不过包含了堆上分配的资源
  // 需要注意的是，这里的tz，包括下面的tzr，行看似是生成Tensor但实际其并没有涉及到全局内存到读写（并没有Tensor被拷贝），
  // 只是利用Layout进行tensor的表达和变换数，数据实体没有移动，只有在copy的时候才有实际的数据读写
  Tensor tz = make_tensor(make_gmem_ptr(z), make_shape(num)); // 8个half的寄存器数组
  Tensor tx = make_tensor(make_gmem_ptr(x), make_shape(num)); // 8个half的寄存器数组
  Tensor ty = make_tensor(make_gmem_ptr(y), make_shape(num)); // 8个half的寄存器数组

  /**
  说一下这段话啥意思，假设num大小为1024
  local tile方法就是对tensor进行分块，分块的大小就是make_shape(Int<kNumElemPerThread>{})参数决定的
  这里Int<kNumElemPerThread>{}等价于Int<8>{}，是一个编译时常量
  这里之所以要用编译时常量，是为了避免运行时常量带来的local mem的问题(注意这里tzr还是指向gmem，下面的tzR才指向寄存器)
  由于kNumElemPerThread值为8，所以local tile相当于对1024长度的数据做分tile，每个tile长度为8
  而make_coord(idx)表示的tile的索引，也就是说当前这个线程的会对第idx个tile的数据进行处理
  */
  Tensor tzr = local_tile(tz, make_shape(Int<kNumElemPerThread>{}), make_coord(idx));
  Tensor txr = local_tile(tx, make_shape(Int<kNumElemPerThread>{}), make_coord(idx));
  Tensor tyr = local_tile(ty, make_shape(Int<kNumElemPerThread>{}), make_coord(idx));

  // 这里使用make tensor like方法创建栈上tensor
  // 栈上tensor的数据一般放在寄存器或者local mem上（具体是寄存器还是local mem由编译器决定）
  Tensor txR = make_tensor_like(txr);
  Tensor tyR = make_tensor_like(tyr);
  Tensor tzR = make_tensor_like(tzr);

  // LDG.128
  // copy行通过调用cute提供的copy函数实现全局内存数据读入到寄存器空间，此处会生成LDG.128指令；
  copy(txr, txR);
  copy(tyr, tyR);

  // 重复系数a、b、c构造half2类型的系数，以利用的HFMA2的指令完成后续计算；
  // {a, a}中的a是传入参数，是z = ax + by + c这个的a
  // 这里使用half2类型，减少half类型引入的PRMT指令的转换和开销
  // 关于为什么half2能避免PRMT的引入，由于篇幅原因，我写在了cutlass.md
  half2 a2 = {a, a};
  half2 b2 = {b, b};
  half2 c2 = {c, c};

  // auto tzR2等行通过recast指令实现连续的half类型到half2类型的转换，以便能利用更高效的HFMA2指令；
  // recast类似于C++中的reinterpret_cast语义
  // recast为half2，变成4个half2
  // 原本tzR的长度为8，有8个half，经过下面操作后长度就变成4了，其中每个元素都是两个half
  auto tzR2 = recast<half2>(tzR);  // 4个half2
  auto txR2 = recast<half2>(txR);
  auto tyR2 = recast<half2>(tyR);

  // z = ax + by + c
  // pragma 及后续for行实现了多个元素的z = ax + by + c的计算，并且通过
  // 括号将该计算通过两个HFMA2指令实现，如果没有括号，则其会
  // 生成 HMUL2 + HMUL2 + HADD2 + HADD2指令（由于乘法不满足结合律，且IEEE
  // 规定了浮点数计算的顺序需按照代码书写顺序）
#pragma unroll
  for (int i = 0; i < size(tzR2); ++i) {
    // two hfma2 instruction
    tzR2(i) = txR2(i) * a2 + (tyR2(i) * b2 + c2);
  }

  auto tzRx = recast<half>(tzR2);

  // STG.128
  copy(tzRx, tzr);
}