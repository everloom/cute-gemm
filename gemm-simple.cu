#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdlib>
#include <cmath>
#include <cute/tensor.hpp>

template <typename T>
void gen_rand_data(T *data, int n);

// 注意，这个代码没有使用smem，是直接从gmem读数据到reg进行计算，然后结果写回gmem
// 注意，这个代码没有使用smem，是直接从gmem读数据到reg进行计算，然后结果写回gmem
// 注意，这个代码没有使用smem，是直接从gmem读数据到reg进行计算，然后结果写回gmem
// 注意，这个代码没有使用smem，是直接从gmem读数据到reg进行计算，然后结果写回gmem
// 当然，这里还是使用的分块矩阵乘
template <typename T, int kTileM, int kTileN, int kTileK, typename TiledMMA>
__global__ void gemm_simple(T *Cptr, const T *Aptr, const T *Bptr, int m, int n, int k) {

  using namespace cute;

  // 为了编译时决策和优化，我们把stride中的连续维度1表示为编译时常量的
  // 形式，即Int<1>{}，这样后续对矩阵进行操作的时候如果需要
  // 用到stride的计算则可以利用编译时的决策和优化减少不必要的运行时运算。

  // 这里需要注意B的shape。这个代码中A和C是行主序，B是列主序，shape (k, n), stride (1, k)
  // 但reed说为了后续的循环的时候可以写成reduce的形式
  // 所以将这里B的layout变为了 shape (n, k), stride (k, 1)，相当于对原始的B做了转置
  Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k), make_stride(k, Int<1>{}));
  Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k), make_stride(k, Int<1>{}));
  Tensor C = make_tensor(make_gmem_ptr(Cptr), make_shape(m, n), make_stride(n, Int<1>{}));

  int ix = blockIdx.x;
  int iy = blockIdx.y;

  // 在进行Tensor分块的时候也将编译时能确定的量进行了Int<>化，用以指示该维度
  // 信息是编译时常量，编译器可以在编译阶段完成必要的路径决策和优化计算，避免运行时的开销

  // 问了下cursor，说这里的make tile方法是定义了分块的形状信息
  // 以gA为例，A的原本的形状是m*k = 81920*256，
  // make tile的分块形状就是kTileM*kTileK（256*32），相当于tile大小是256*32
  // make_coord(iy, _)表示取出对gA分tile之后的第iy行上的所有tile（_表示指定所有列）
  // gA的shape是(kTileM, kTileK, num_tile_k) (128, 32, 8)，128x32是分块的大小，8是因为make_coord取了k维度上的所有分块（256/32=8）
  // gB的shape是(kTileN, kTileK, num_tile_k) (128, 32, 8)
  // gc的shape是(kTileM, kTileN) (128, 128)
  Tensor gA = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(iy, _));
  Tensor gB = local_tile(B, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(ix, _));
  Tensor gC = local_tile(C, make_tile(Int<kTileM>{}, Int<kTileN>{}), make_coord(iy, ix));
  //  gA(kTileM, kTileK, num_tile_k)
  //  gB(kTileN, kTileK, num_tile_k)
  //  gC(kTileM, kTileN) 

  // 这里TiledMMA类就是main函数里面的那个using MMA，这里实例化的tiled_mma支持一个warp处理总共32x32x16的计算
  TiledMMA tiled_mma;
  /**
  reed的博客说这里的get slice方法能通过线程id获取线程对应的ThrMMA的结构，这个结构描述
  了线程级实现D = A x B + C任务的功能抽象（这是reed博客里面的原话，说的很抽象，我理解反正就是根据线程id拿到了一个class对象就完了，每个线程id都对应了一个ThrMMA对象）
  然后parition_A方法是reed说是对逻辑Tensor针对该线程的划分，这里的逻辑Tensor指的是传入的大的逻辑单元（指的gA）
  然后parition_A的返回值返回该线程需要进行的任务的Tensor描述（即tAgA）（这个也是reed的原话）
  然后tAgA的shape是(MMA, MMA_M, MMA_K, num_tile_k)，MMA表示TiledMMA一次能做的矩阵运算所需要的数据，即32x32x16，
  MMA_M大小为kTileM/32=4(这里32是TiledMMA处理数据的32x32x16的第一个32)，MMA_K大小为kTileK/16=2 (这里16是32x32x16的那个16)，
  num_tile_k表示k上需要迭代多少次，和gA的num_tile_k一致。所以本质上partition_A是对gA的前两维进行划分
  然后partition_fragment_A返回tAgA的一个寄存器对象，其中传入参数gA(_, _, 0)类似于gA[:, :, 0]
  已知gA的shape为(kTileM, kTileK, num_tile_k)，所以gA[:, :, 0]的shape为(kTileM, kTileK)，忽略了num_tile_k
  所以把gA(_, _, 0)作为参数传给partition_fragment_A，返回的tArA相比thr_mma自然也没有num_tile_k这个维度了，只剩下了(MMA, MMA_M, MMA_K)

  这上面的描述都是reed的描述，看起来很抽象
  我举得干脆就这样简单理解，tAgA指向的就是一个gmem数据对象，然后这个数据对象的shape是(MMA_M, MMA_K, num_tile_k) (这里忽略第一维的MMA)
  然后其中的每个元素的大小都是32x16。也就是说tAgA就是一个大小为(MMA_M, MMA_K, num_tile_k)(4,2,8)的gmem矩阵，矩阵中每个元素都是32x16的一个小矩阵（相当于总共128x32x8的大小）
  tBgB和tCgC同理，然后tArA相当于是一个大小为(MMA_M, MMA_K)(4,2)的寄存器矩阵，矩阵中每个元素都是一个小的32x16的矩阵（相当于总共128x32的大小）
  可以看到下面tCgC的shape是（4, 4)，其中每个元素是一个32x32的小矩阵，所以总共就是128x128，刚好就是threadblock tile的大小
  */
  auto thr_mma = tiled_mma.get_slice(threadIdx.x);
  auto tAgA = thr_mma.partition_A(gA);  // (MMA, MMA_M, MMA_K, num_tile_k)，(MMA, 4, 2, 8)，其中每个元素是一个32x16的小矩阵
  auto tBgB = thr_mma.partition_B(gB);  // (MMA, MMA_N, MMA_K, num_tile_k)，(MMA, 4, 2, 8)，其中每个元素是一个32x16的小矩阵
  auto tCgC = thr_mma.partition_C(gC);  // (MMA, MMA_M, MMA_N) (MMA, 4, 4)，其中每个元素是一个32x32的小矩阵，所以总共就是128x128，刚好就是threadblock tile的大小

  auto tArA = thr_mma.partition_fragment_A(gA(_, _, 0));  // (MMA, MMA_M, MMA_K) (MMA, 4, 2)
  auto tBrB = thr_mma.partition_fragment_B(gB(_, _, 0));  // (MMA, MMA_N, MMA_K) (MMA, 4, 2)
  auto tCrC = thr_mma.partition_fragment_C(gC(_, _));     // (MMA, MMA_M, MMA_N) (MMA, 4, 4)
  // 对tCrC初始化为0
  clear(tCrC);
  // 这里相当于gA.shape[2]，返回的就是num_tile_k
  int num_tile_k = size<2>(gA); // 这里值为8
#pragma unroll 1
  for(int itile = 0; itile < num_tile_k; ++itile) {
    // 将gmem的数据拷贝到reg中
    // cute::copy在不指定Copy_Atom时采用UniversalCopy实现，即简单的cuda语言层面的T d = s形式
    cute::copy(tAgA(_, _, _, itile), tArA);
    cute::copy(tBgB(_, _, _, itile), tBrB);
    // 执行mma，warp level执行
    cute::gemm(tiled_mma, tCrC, tArA, tBrB, tCrC);
  }

  // 将reg的结果拷贝回gmem
  cute::copy(tCrC, tCgC); 
}

int main() {
  srand(10086);

  using T = cute::half_t;
  using namespace cute;

  T *Cptr;
  T *Aptr;
  T *Bptr;

  int m = 81920;
  int n = 256;
  int k = 256;

  cudaMalloc(&Cptr, sizeof(T) * m * n);
  cudaMalloc(&Aptr, sizeof(T) * m * k);
  cudaMalloc(&Bptr, sizeof(T) * k * n);

  T *Aptr_host;
  T *Bptr_host;
  Aptr_host = (T*)malloc(sizeof(T) * m * k);
  Bptr_host = (T*)malloc(sizeof(T) * n * k);
  gen_rand_data(Aptr_host, m * k);
  gen_rand_data(Bptr_host, n * k);

  cudaMemcpy(Aptr, Aptr_host, sizeof(T) * m * k, cudaMemcpyHostToDevice);
  cudaMemcpy(Bptr, Bptr_host, sizeof(T) * n * k, cudaMemcpyHostToDevice);
  
  // SM80_16x8x16_F16F16F16F16_TN表示架构是80，使用16x8x16的Tensor Core矩阵乘法指令
  // 数据精度和计算精度都为fp16，TN表示a是行主序，b是列主序
  // N代表normal，由于cublas默认列主序，所以normal也代表列主序；T代表transpose，也就是行主序
  // 然后那个MMA_Traits和MMA_Atom我猜应该就是api固定用法，所以没仔细研究，对这俩感兴趣的话看reed的这个博客https://zhuanlan.zhihu.com/p/663092747
  using mma_op = SM80_16x8x16_F16F16F16F16_TN;
  using mma_traits = MMA_Traits<mma_op>;
  using mma_atom = MMA_Atom<mma_traits>;

  /*
  理解这里卡了我好久，我现在把我目前理解到的写在这里
  先说结论，这里MMA类负责处理总共32x32x16的数据的计算（MxN维度处理32x32，K维度是16）
  首先需要注意的是，这里的分块矩阵乘没有使用smem，所以不存在warp tile的概念，只有threadblock tile的概念
  （只有在使用了smem之后才有warp tile这个分tile级别）
  这里可以看到两个make layout的参数，根据reed的cute之Tensor的定义https://zhuanlan.zhihu.com/p/663092747
  可以看到，第一个make_layout(Shape<_2, _2, _1>{})对应了TiledMMA的AtomLayoutMNK参数
  第二个make_layout(Shape<_1, _2, _1>{})对应了TiledMMA的ValLayoutMNK
  然后这里解释一下这两个参数是什么意思
  首先第一个layout的2 2 1，分别表示在m n k方向扩展mma的shape，这会改变block的大小
  假设原本一个block中只有一个warp执行一个mma，再经过这个layout为2 2 1的AtomLayoutMNK参数
  变换之后，这里mma的m维度需要扩展2，n维度需要扩展2，k维度扩展1
  经过AtomLayoutMNK变换之后的block中的线程分布如下所示，可以看到一个block从一个warp执行mma变成了4个warp执行mma
  变换之后using MMA的这个MMA类就负责处理总共32x8x16的计算

    ┌──────────────┬──────────────┐
    │ Warp 0       │ Warp 1       │ 
    │ mma 16×8x16  │ mma 16×8x16  │
    ├──────────────┼──────────────┤ 
    │ Warp 2       │ Warp 3       │
    │ mma 16×8x16  │ mma 16×8x16  │
    └──────────────┴──────────────┘ 

  然后是make_layout(Shape<_1, _2, _1>{})的ValLayoutMNK参数，这个参数只会影响MMA负责处理的数据
  而不会影响一个block中thread的数量，具体来说就是让每个线程在某个维度上重复处理数据，使得MMA这个类
  在不改变block大小的前提下增加处理数据的数量。这里的layout只在N维度上为2，在M和K维度上为1，
  所以相当于在N维度上重复处理2个数，所以原本MMA类处理32x8x16的数据，经过ValLayoutMNK的调整之后，
  处理的数据量就变为了32x32x16
  */
  using MMA = decltype(make_tiled_mma(mma_atom{}, 
                      make_layout(Shape<_2, _2, _1>{}), 
                      make_layout(Shape<_1, _2, _1>{})));
  constexpr int kTileM = 128; 
  constexpr int kTileN = 128; 
  constexpr int kTileK = 32; 

  dim3 block(size(MMA{})); // size(MMA{})为128，对应了上面MMA里面的注释说的，一个block是4个warp
  dim3 grid(n / kTileN, m / kTileM);
  for (int i = 0; i < 100; ++i) {
    gemm_simple<T, kTileM, kTileN, kTileK, MMA><<<grid, block>>>(Cptr, Aptr, Bptr, m, n, k);
  }
  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  printf("err = %d, str = %s\n", err, cudaGetErrorString(err));

  // cublas
  T *Cptr_cublas;

  cudaMalloc(&Cptr_cublas, sizeof(T) * m * n);

  cublasHandle_t handle;
  cublasCreate(&handle);

  half alpha = half(1.f);
  half beta = half(0.f);
  for (int i = 0; i < 100; ++i) {
    cublasStatus_t ret = cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
          	  n, m, k,
          	  &alpha,
          	  (half *)Bptr, k,
          	  (half *)Aptr, k,
          	  &beta,
          	  (half *)Cptr_cublas, n);
    if (ret != CUBLAS_STATUS_SUCCESS) {
      printf("blas err = %d, str = %s\n", ret, cublasGetStatusString(ret));
    }
  }

  cudaDeviceSynchronize();
  err = cudaGetLastError();
  printf("err = %d, str = %s\n", err, cudaGetErrorString(err));

  T *Cptr_host;
  T *Cptr_cublas_host;

  Cptr_host = (T*)malloc(sizeof(T) * m * n);
  Cptr_cublas_host = (T*)malloc(sizeof(T) * m * n);

  // compare
  cudaMemcpy(Cptr_host, Cptr, sizeof(T) * m * n, cudaMemcpyDeviceToHost);
  cudaMemcpy(Cptr_cublas_host, Cptr_cublas, sizeof(T) * m * n, cudaMemcpyDeviceToHost);

  float threshold = 0.1;
  for (int i = 0; i < m * n; ++i) {
    float v1 = Cptr_host[i];
    float v2 = Cptr_cublas_host[i];
    if (fabs(v2 - v1) > threshold) {
      printf("v1 = %f, v2 = %f\n", v1, v2);
    }
  }

  Tensor tensor_C = make_tensor(Cptr_host, make_shape(m, n), make_stride(n, 1));
  Tensor tensor_C_cublas = make_tensor(Cptr_cublas_host, make_shape(m, n), make_stride(n, 1));

  auto tile = make_tile(8, 8);
  auto coor = make_coord(0, 0);
  Tensor tc1 = local_tile(tensor_C, tile, coor);
  Tensor tc1_cublas = local_tile(tensor_C_cublas, tile, coor);

  print_tensor(tc1);
  print_tensor(tc1_cublas);
}

template <typename T>
void gen_rand_data(T *data, int n) {
  for (int i = 0; i < n; ++i) {
    float v = (rand() % 200 - 100) * 0.01;
    data[i] = v;
  }
}
