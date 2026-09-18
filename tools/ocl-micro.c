// ocl-micro.c - OpenCL microbenchmarks used to characterise the BC-250's
// compute path (rusticl -> radeonsi -> amdgpu, gfx1013). See OPENCL-PERF.md.
//
//   gcc -O2 -o ocl-micro tools/ocl-micro.c -lOpenCL
//   RUSTICL_ENABLE=radeonsi ./ocl-micro [test]
//
// test: all (default) | info | alu | bw | cache | lds | atomics | lat | tile | occ |
//       launch | xfer
//
// Every number it prints is a best-of-N wall-clock measurement around
// clEnqueueNDRangeKernel + clFinish, so it includes dispatch overhead. The
// point is not absolute peak, it is the ratio between things that should cost
// the same and don't.
//
// MIT License, Copyright (c) 2026 mxreyer.
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+1e-9*t.tv_nsec; }
#define CHK(x) do{ cl_int _e=(x); if(_e!=CL_SUCCESS){ fprintf(stderr,"OpenCL error %d at line %d\n",_e,__LINE__); exit(1);} }while(0)

static cl_device_id  g_dev;
static cl_context    g_ctx;
static cl_command_queue g_q;
static cl_uint       g_cus;

static cl_program build(const char *src,const char *opts)
{
    cl_int err;
    cl_program p = clCreateProgramWithSource(g_ctx,1,&src,0,&err); CHK(err);
    if(clBuildProgram(p,1,&g_dev,opts?opts:"",0,0)!=CL_SUCCESS){
        size_t n; clGetProgramBuildInfo(p,g_dev,CL_PROGRAM_BUILD_LOG,0,0,&n);
        char *log=malloc(n+1); clGetProgramBuildInfo(p,g_dev,CL_PROGRAM_BUILD_LOG,n,log,0);
        fprintf(stderr,"build failed:\n%s\n",log); exit(1);
    }
    return p;
}

// best-of-reps wall time for one NDRange
static double timeit(cl_kernel k,size_t g,size_t l,int reps)
{
    CHK(clEnqueueNDRangeKernel(g_q,k,1,0,&g,&l,0,0,0)); CHK(clFinish(g_q)); // warm
    double best=1e30;
    for(int r=0;r<reps;r++){
        double t0=now();
        CHK(clEnqueueNDRangeKernel(g_q,k,1,0,&g,&l,0,0,0));
        CHK(clFinish(g_q));
        double d=now()-t0; if(d<best) best=d;
    }
    return best;
}

// ---------------------------------------------------------------- info
static void t_info(void)
{
    char name[256]={0},ver[128]={0},prof[64]={0};
    cl_uint mhz,cline; cl_ulong lsz,csz,gsz,alloc; size_t wg,wgm;
    cl_device_local_mem_type lt; cl_device_mem_cache_type ct;
    clGetDeviceInfo(g_dev,CL_DEVICE_NAME,sizeof name,name,0);
    clGetDeviceInfo(g_dev,CL_DRIVER_VERSION,sizeof ver,ver,0);
    clGetDeviceInfo(g_dev,CL_DEVICE_PROFILE,sizeof prof,prof,0);
    clGetDeviceInfo(g_dev,CL_DEVICE_MAX_CLOCK_FREQUENCY,sizeof mhz,&mhz,0);
    clGetDeviceInfo(g_dev,CL_DEVICE_MAX_WORK_GROUP_SIZE,sizeof wg,&wg,0);
    clGetDeviceInfo(g_dev,CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,0,0,0); // ignored
    clGetDeviceInfo(g_dev,CL_DEVICE_LOCAL_MEM_TYPE,sizeof lt,&lt,0);
    clGetDeviceInfo(g_dev,CL_DEVICE_LOCAL_MEM_SIZE,sizeof lsz,&lsz,0);
    clGetDeviceInfo(g_dev,CL_DEVICE_GLOBAL_MEM_CACHE_TYPE,sizeof ct,&ct,0);
    clGetDeviceInfo(g_dev,CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,sizeof csz,&csz,0);
    clGetDeviceInfo(g_dev,CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE,sizeof cline,&cline,0);
    clGetDeviceInfo(g_dev,CL_DEVICE_GLOBAL_MEM_SIZE,sizeof gsz,&gsz,0);
    clGetDeviceInfo(g_dev,CL_DEVICE_MAX_MEM_ALLOC_SIZE,sizeof alloc,&alloc,0);
    wgm = 0;
    printf("device                 %s\n",name);
    printf("driver                 %s   profile %s\n",ver,prof);
    printf("compute units          %u @ %u MHz  -> %.2f TFLOP/s fp32 theoretical\n",
           g_cus,mhz,(double)g_cus*64*2*mhz/1e6);
    printf("max work-group         %zu\n",wg);
    printf("global mem             %.2f GiB   max alloc %.2f GiB\n",gsz/1073741824.0,alloc/1073741824.0);
    printf("LOCAL_MEM_TYPE         %s   size %llu\n",
           lt==CL_LOCAL?"CL_LOCAL":(lt==CL_GLOBAL?"CL_GLOBAL  <-- see 'lds' test":"CL_NONE"),
           (unsigned long long)lsz);
    printf("GLOBAL_MEM_CACHE_TYPE  %s   size %llu  cacheline %u %s\n",
           ct==CL_NONE?"CL_NONE":(ct==CL_READ_ONLY_CACHE?"READ_ONLY":"READ_WRITE"),
           (unsigned long long)csz,cline, ct==CL_NONE?" <-- see 'cache' test":"");
    (void)wgm;
}

// ---------------------------------------------------------------- alu
// fma() vs mad() vs a*b+c. On rusticl these should be the same instruction.
static const char *ALU_TMPL =
"__kernel void bench(__global float *out, float a, int iters){\n"
"  float x0=a,x1=a+1.f,x2=a+2.f,x3=a+3.f,x4=a+4.f,x5=a+5.f,x6=a+6.f,x7=a+7.f;\n"
"  float b=1.0000001f, c=0.9999999f; (void)b;(void)c;\n"
"  for(int i=0;i<iters;i++){ OPX(x0) OPX(x1) OPX(x2) OPX(x3) OPX(x4) OPX(x5) OPX(x6) OPX(x7) }\n"
"  float s=x0+x1+x2+x3+x4+x5+x6+x7;\n"
"  if(s==12345.678f) out[get_global_id(0)]=s;\n}\n";

static void t_alu(void)
{
    struct { const char *name,*op; int flops; } c[] = {
        {"fma(x,b,c)",  "#define OPX(v) v=fma(v,b,c);\n", 2},
        {"mad(x,b,c)",  "#define OPX(v) v=mad(v,b,c);\n", 2},
        {"x*b+c",       "#define OPX(v) v=v*b+c;\n",      2},
        {"x*b  (mul)",  "#define OPX(v) v=v*b;\n",        1},
        {"x+c  (add)",  "#define OPX(v) v=v+c;\n",        1},
    };
    cl_int err;
    cl_mem out=clCreateBuffer(g_ctx,CL_MEM_WRITE_ONLY,1<<20,0,&err); CHK(err);
    printf("  %-14s %10s %12s\n","op","ms","GFLOP/s");
    for(unsigned i=0;i<sizeof c/sizeof*c;i++){
        char src[8192]; snprintf(src,sizeof src,"%s%s",c[i].op,ALU_TMPL);
        cl_program p=build(src,"-cl-mad-enable -cl-fast-relaxed-math");
        cl_kernel k=clCreateKernel(p,"bench",&err); CHK(err);
        float a=1.f; int iters=20000; size_t l=256,g=(size_t)g_cus*8*l;
        CHK(clSetKernelArg(k,0,sizeof out,&out));
        CHK(clSetKernelArg(k,1,sizeof a,&a));
        CHK(clSetKernelArg(k,2,sizeof iters,&iters));
        double t=timeit(k,g,l,5);
        printf("  %-14s %10.2f %12.1f\n",c[i].name,t*1e3,(double)g*iters*8.0*c[i].flops/t/1e9);
        clReleaseKernel(k); clReleaseProgram(p);
    }
    clReleaseMemObject(out);
}

// ---------------------------------------------------------------- bw / cache / lds
static const char *MEM_SRC =
"__kernel void rd(__global const float4 *in,__global float *out,int n4,int reps){\n"
"  int gid=get_global_id(0),stride=get_global_size(0); float4 acc=(float4)(0.f);\n"
"  for(int r=0;r<reps;r++) for(int i=gid;i<n4;i+=stride) acc+=in[i];\n"
"  float s=acc.x+acc.y+acc.z+acc.w; if(s==12345.678f) out[gid]=s;\n}\n"
"__kernel void wr(__global float4 *out,int n4){\n"
"  int gid=get_global_id(0),stride=get_global_size(0); float4 v=(float4)((float)gid);\n"
"  for(int i=gid;i<n4;i+=stride) out[i]=v;\n}\n"
"__kernel void cp(__global const float4 *in,__global float4 *out,int n4){\n"
"  int gid=get_global_id(0),stride=get_global_size(0);\n"
"  for(int i=gid;i<n4;i+=stride) out[i]=in[i];\n}\n"
"__kernel void lds(__global float *out,int iters){\n"
"  __local float buf[4096]; int lid=get_local_id(0),ls=get_local_size(0);\n"
"  for(int i=lid;i<4096;i+=ls) buf[i]=(float)i;\n"
"  barrier(CLK_LOCAL_MEM_FENCE);\n"
"  float acc=0.f; int idx=lid;\n"
"  for(int it=0;it<iters;it++){ acc+=buf[idx&4095]; idx+=ls; acc+=buf[(idx+7)&4095]; idx+=ls;\n"
"    acc+=buf[(idx+13)&4095]; idx+=ls; acc+=buf[(idx+29)&4095]; idx+=ls; }\n"
"  if(acc==12345.678f) out[get_global_id(0)]=acc;\n}\n";

static void t_bw(void)
{
    cl_int err; size_t bytes=512ull<<20; int n4=(int)(bytes/16), one=1;
    cl_program p=build(MEM_SRC,"");
    cl_mem A=clCreateBuffer(g_ctx,CL_MEM_READ_WRITE,bytes,0,&err); CHK(err);
    cl_mem B=clCreateBuffer(g_ctx,CL_MEM_READ_WRITE,bytes,0,&err); CHK(err);
    cl_mem O=clCreateBuffer(g_ctx,CL_MEM_WRITE_ONLY,1<<20,0,&err); CHK(err);
    float z=0; CHK(clEnqueueFillBuffer(g_q,A,&z,4,0,bytes,0,0,0)); CHK(clFinish(g_q));
    size_t l=256,g=(size_t)g_cus*16*l;
    struct { const char *nm,*kn; int mult; } t[]={{"read","rd",1},{"write","wr",1},{"copy","cp",2}};
    for(int i=0;i<3;i++){
        cl_kernel k=clCreateKernel(p,t[i].kn,&err); CHK(err);
        if(i==0){ CHK(clSetKernelArg(k,0,sizeof A,&A)); CHK(clSetKernelArg(k,1,sizeof O,&O));
                  CHK(clSetKernelArg(k,2,sizeof n4,&n4)); CHK(clSetKernelArg(k,3,sizeof one,&one)); }
        else if(i==1){ CHK(clSetKernelArg(k,0,sizeof A,&A)); CHK(clSetKernelArg(k,1,sizeof n4,&n4)); }
        else { CHK(clSetKernelArg(k,0,sizeof A,&A)); CHK(clSetKernelArg(k,1,sizeof B,&B)); CHK(clSetKernelArg(k,2,sizeof n4,&n4)); }
        double d=timeit(k,g,l,5);
        printf("  %-6s %8.2f GB/s   (%zu MiB)\n",t[i].nm,(double)bytes*t[i].mult/d/1e9,bytes>>20);
        clReleaseKernel(k);
    }
    clReleaseMemObject(A);clReleaseMemObject(B);clReleaseMemObject(O);clReleaseProgram(p);
}

static void t_cache(void)
{
    cl_int err; size_t maxb=256ull<<20;
    cl_program p=build(MEM_SRC,"");
    cl_kernel k=clCreateKernel(p,"rd",&err); CHK(err);
    cl_mem A=clCreateBuffer(g_ctx,CL_MEM_READ_ONLY,maxb,0,&err); CHK(err);
    cl_mem O=clCreateBuffer(g_ctx,CL_MEM_WRITE_ONLY,1<<20,0,&err); CHK(err);
    float z=1; CHK(clEnqueueFillBuffer(g_q,A,&z,4,0,maxb,0,0,0)); CHK(clFinish(g_q));
    printf("  working set -> read bandwidth (a plateau above DRAM speed IS a cache)\n");
    for(size_t kb=32; kb<=(maxb>>10); kb*=2){
        size_t bytes=kb<<10; int n4=(int)(bytes/16);
        int reps=(int)(268435456ull/bytes); if(reps<1)reps=1; if(reps>4096)reps=4096;
        size_t l=256,g=(size_t)g_cus*8*l; if(g>(size_t)n4) g=(size_t)((n4+255)/256)*256;
        CHK(clSetKernelArg(k,0,sizeof A,&A)); CHK(clSetKernelArg(k,1,sizeof O,&O));
        CHK(clSetKernelArg(k,2,sizeof n4,&n4)); CHK(clSetKernelArg(k,3,sizeof reps,&reps));
        double d=timeit(k,g,l,3);
        printf("  %8zu KiB %10.1f GB/s\n",kb,(double)bytes*reps/d/1e9);
    }
    clReleaseMemObject(A);clReleaseMemObject(O);clReleaseKernel(k);clReleaseProgram(p);
}

static void t_lds(void)
{
    cl_int err; cl_program p=build(MEM_SRC,"");
    cl_kernel k=clCreateKernel(p,"lds",&err); CHK(err);
    cl_mem O=clCreateBuffer(g_ctx,CL_MEM_WRITE_ONLY,1<<20,0,&err); CHK(err);
    int iters=4000; size_t l=256,g=(size_t)g_cus*4*l;
    CHK(clSetKernelArg(k,0,sizeof O,&O)); CHK(clSetKernelArg(k,1,sizeof iters,&iters));
    double d=timeit(k,g,l,5);
    printf("  __local read %.0f GB/s\n",(double)g*iters*4.0*4.0/d/1e9);
    printf("  (compare with the 'bw' read figure: if __local is much faster, it is\n"
           "   real on-chip LDS, whatever CL_DEVICE_LOCAL_MEM_TYPE claims)\n");
    clReleaseMemObject(O);clReleaseKernel(k);clReleaseProgram(p);
}

// ---------------------------------------------------------------- atomics
static const char *ATOM_SRC =
"void atomic_addf(__global volatile float *ptr,float v){\n"     // dlprimitives' fallback
"  float oldv=*ptr;\n"
"  for(;;){ float newv=oldv+v;\n"
"    int prev=atomic_cmpxchg((__global volatile int*)(ptr),as_int(oldv),as_int(newv));\n"
"    if(prev==as_int(oldv)) return; oldv=as_float(prev); }\n}\n"
"__kernel void plain (__global float *o,int n,int reps){ int g=get_global_id(0);\n"
"  for(int r=0;r<reps;r++) o[(g*4+r)%n]=1.0f; }\n"
"__kernel void casadd(__global float *o,int n,int reps){ int g=get_global_id(0);\n"
"  for(int r=0;r<reps;r++) atomic_addf(&o[(g*4+r)%n],1.0f); }\n"
"__kernel void intadd(__global int *o,int n,int reps){ int g=get_global_id(0);\n"
"  for(int r=0;r<reps;r++) atomic_add(&o[(g*4+r)%n],1); }\n";

static void t_atomics(void)
{
    cl_int err; cl_program p=build(ATOM_SRC,"");
    size_t slots=64u<<20;
    cl_mem O=clCreateBuffer(g_ctx,CL_MEM_READ_WRITE,slots*4,0,&err); CHK(err);
    float z=0; CHK(clEnqueueFillBuffer(g_q,O,&z,4,0,slots*4,0,0,0)); CHK(clFinish(g_q));
    int reps=64; size_t l=256,g=(size_t)g_cus*8*l;
    int spread[]={(int)slots,1<<16,4096};
    const char *sl[]={"uncontended (64M slots)","64K slots","4K slots (heavy contention)"};
    const char *kn[]={"plain","casadd","intadd"};
    const char *nm[]={"plain store","CAS-loop float add","native int atomic add"};
    for(int s=0;s<3;s++){
        printf("  %s:\n",sl[s]);
        for(int i=0;i<3;i++){
            cl_kernel k=clCreateKernel(p,kn[i],&err); CHK(err);
            CHK(clSetKernelArg(k,0,sizeof O,&O)); CHK(clSetKernelArg(k,1,sizeof spread[s],&spread[s]));
            CHK(clSetKernelArg(k,2,sizeof reps,&reps));
            double d=timeit(k,g,l,5);
            printf("    %-24s %8.2f Gop/s\n",nm[i],(double)g*reps/d/1e9);
            clReleaseKernel(k);
        }
    }
    clReleaseMemObject(O); clReleaseProgram(p);
}

// ---------------------------------------------------------------- lat
// One thread chasing a pointer with a 4 KiB stride: each load must complete
// before the next address is known, so this is raw dependent-load latency.
static void t_lat(void)
{
    cl_int err; int N=1<<22;
    int *h=malloc((size_t)N*4);
    for(int i=0;i<N;i++) h[i]=(int)(((long)i+1024)%N);
    cl_program p=build(
        "__kernel void chase(__global const int *in,__global int *out,int steps){\n"
        "  int q=0; for(int i=0;i<steps;i++) q=in[q];\n"
        "  out[get_global_id(0)]=q;\n}\n","");
    cl_kernel k=clCreateKernel(p,"chase",&err); CHK(err);
    cl_mem A=clCreateBuffer(g_ctx,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,(size_t)N*4,h,&err); CHK(err);
    cl_mem O=clCreateBuffer(g_ctx,CL_MEM_WRITE_ONLY,4096,0,&err); CHK(err);
    int steps=20000;
    CHK(clSetKernelArg(k,0,sizeof A,&A)); CHK(clSetKernelArg(k,1,sizeof O,&O));
    CHK(clSetKernelArg(k,2,sizeof steps,&steps));
    double d=timeit(k,1,1,3);
    printf("  dependent global load  %.1f ns\n",d/steps*1e9);
    free(h); clReleaseMemObject(A); clReleaseMemObject(O);
    clReleaseKernel(k); clReleaseProgram(p);
}


// ---------------------------------------------------------------- tile
// The roofline that actually binds a tiled GEMM/convolution here. An RxR
// register tile does R*R FMAs per 2R operand reads from __local, i.e. R/2
// FMAs per LDS float. Sweeping R walks from "LDS-bandwidth bound" into
// "the register allocator gave up".
static const char *TILE_SRC =
"#define R RTILE\n"
"__kernel void pure_fma(__global float *out,int iters){\n"
"  float c[R][R];\n"
"  #pragma unroll\n for(int i=0;i<R;i++){\n  #pragma unroll\n for(int j=0;j<R;j++) c[i][j]=(float)(i+j);}\n"
"  float a=1.0000001f,b=0.9999999f;\n"
"  for(int k=0;k<iters;k++){\n"
"    #pragma unroll\n for(int i=0;i<R;i++){\n"
"      #pragma unroll\n for(int j=0;j<R;j++) c[i][j]=mad(a,b,c[i][j]);\n    }\n"
"    a+=1e-9f;\n  }\n"
"  float s=0;\n #pragma unroll\n for(int i=0;i<R;i++){\n #pragma unroll\n for(int j=0;j<R;j++) s+=c[i][j];}\n"
"  if(s==12345.678f) out[get_global_id(0)]=s;\n}\n"
"__kernel void lds_fma(__global float *out,int iters){\n"
"  __local float tile[2048];\n"
"  int lid=get_local_id(0),ls=get_local_size(0);\n"
"  for(int i=lid;i<2048;i+=ls) tile[i]=(float)i*1e-6f;\n"
"  barrier(CLK_LOCAL_MEM_FENCE);\n"
"  float c[R][R];\n"
"  #pragma unroll\n for(int i=0;i<R;i++){\n #pragma unroll\n for(int j=0;j<R;j++) c[i][j]=0.f;}\n"
"  int base=lid*R;\n"
"  for(int k=0;k<iters;k++){\n"
"    float av[R],bv[R];\n"
"    #pragma unroll\n for(int i=0;i<R;i++) av[i]=tile[(base+i+k)&2047];\n"
"    #pragma unroll\n for(int j=0;j<R;j++) bv[j]=tile[(base+j+k+1024)&2047];\n"
"    #pragma unroll\n for(int i=0;i<R;i++){\n"
"      #pragma unroll\n for(int j=0;j<R;j++) c[i][j]=mad(av[i],bv[j],c[i][j]);\n    }\n"
"  }\n"
"  float s=0;\n #pragma unroll\n for(int i=0;i<R;i++){\n #pragma unroll\n for(int j=0;j<R;j++) s+=c[i][j];}\n"
"  if(s==12345.678f) out[get_global_id(0)]=s;\n}\n";

static void t_tile(void)
{
    cl_int err;
    cl_mem O=clCreateBuffer(g_ctx,CL_MEM_WRITE_ONLY,1<<20,0,&err); CHK(err);
    // measure the LDS read roof first, in float-reads/s
    double reads_per_s;
    {
        cl_program p=build(MEM_SRC,"");
        cl_kernel k=clCreateKernel(p,"lds",&err); CHK(err);
        int iters=4000; size_t l=256,g=(size_t)g_cus*4*l;
        CHK(clSetKernelArg(k,0,sizeof O,&O)); CHK(clSetKernelArg(k,1,sizeof iters,&iters));
        double d=timeit(k,g,l,5);
        reads_per_s=(double)g*iters*4.0/d;
        printf("  LDS roof %.0f G float-reads/s -> an RxR tile caps at %.2f*R TFLOP/s\n\n",
               reads_per_s/1e9, reads_per_s*2/2/1e12);
        clReleaseKernel(k); clReleaseProgram(p);
    }
    printf("  %2s %10s %10s %9s %10s %9s %9s\n",
           "R","pure FMA","+LDS","LDS roof","%% of roof","fma scr","lds scr");
    for(int R=4;R<=13;R++){
        char opts[64]; snprintf(opts,sizeof opts,"-DRTILE=%d -cl-mad-enable",R);
        cl_program p=build(TILE_SRC,opts);
        double gf[2]; cl_ulong scratch[2]={0,0};
        const char *kn[2]={"pure_fma","lds_fma"};
        for(int i=0;i<2;i++){
            cl_kernel k=clCreateKernel(p,kn[i],&err); CHK(err);
            int iters=4000; size_t l=256,g=(size_t)g_cus*4*l;
            CHK(clSetKernelArg(k,0,sizeof O,&O)); CHK(clSetKernelArg(k,1,sizeof iters,&iters));
            clGetKernelWorkGroupInfo(k,g_dev,CL_KERNEL_PRIVATE_MEM_SIZE,sizeof scratch[i],&scratch[i],0);
            double d=timeit(k,g,l,5);
            gf[i]=(double)g*iters*R*R*2.0/d/1e9;
            clReleaseKernel(k);
        }
        double roof=reads_per_s*R/1e9;
        printf("  %2d %10.0f %10.0f %9.0f %9.0f%% %9llu %9llu%s\n",
               R,gf[0],gf[1],roof,100*gf[1]/roof,
               (unsigned long long)scratch[0],(unsigned long long)scratch[1],
               (R&(R-1))==0?"   <- power-of-2 stride: LDS bank conflicts":"");
        clReleaseProgram(p);
    }
    printf("\n  Read it as: while %% of roof stays near 100, LDS bandwidth is the limit.\n"
           "  When a scratch column goes non-zero the register allocator has given\n"
           "  up and is spilling, and that row's throughput collapses. Rows where R\n"
           "  is a power of two read low for an unrelated reason: this kernel indexes\n"
           "  __local at stride R, so those hit LDS bank conflicts. A real GEMM pads\n"
           "  its tiles to avoid that; dlprimitives does (TILE_OFFSET).\n");
    clReleaseMemObject(O);
}

// ---------------------------------------------------------------- occupancy
// How much __local a work-group may hold before it costs residency. The
// kernel is latency-bound (a dependent global pointer chase), so more
// work-groups resident on a CU means more loads in flight and a proportionally
// shorter run. Sweeping the __local allocation walks down the residency steps,
// and the step positions give the LDS budget a CU actually partitions -
// CL_DEVICE_LOCAL_MEM_SIZE only reports the per-work-group maximum.
static const char *OCC_SRC =
"__kernel __attribute__((reqd_work_group_size(256,1,1)))\n"
"void occ(__global const int *chain,__global int *out,int steps){\n"
"  __local int buf[LDS_INTS];\n"
"  int l=get_local_id(0);\n"
"  buf[l]=l;\n"
"  barrier(CLK_LOCAL_MEM_FENCE);\n"
"  int q=buf[l]+get_group_id(0)*257;\n"
"  for(int i=0;i<steps;i++) q=chain[q];\n"
"  buf[l&(LDS_INTS-1)]=q;\n"
"  barrier(CLK_LOCAL_MEM_FENCE);\n"
"  if(q==0x7fffffff) out[get_group_id(0)]=buf[l];\n}\n";

static void t_occ(void)
{
    cl_int err; int N=1<<22;
    int *h=malloc((size_t)N*4);
    for(int i=0;i<N;i++) h[i]=(int)(((long)i+1024)%N);
    cl_mem A=clCreateBuffer(g_ctx,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,(size_t)N*4,h,&err); CHK(err);
    cl_mem O=clCreateBuffer(g_ctx,CL_MEM_WRITE_ONLY,1<<20,0,&err); CHK(err);
    // one work-group per CU: nothing shares a CU, so this is the floor
    double t1=0;
    const int kib[]={4,8,12,16,20,24,28,32,36,40,44,48,56,64};
    printf("  %7s %10s %10s %8s\n","__local","us/round","speedup","implied");
    for(unsigned i=0;i<sizeof kib/sizeof*kib;i++){
        char opts[64]; snprintf(opts,sizeof opts,"-DLDS_INTS=%d",kib[i]*256);
        cl_program p=build(OCC_SRC,opts);
        cl_kernel k=clCreateKernel(p,"occ",&err); CHK(err);
        int steps=3000; size_t l=256;
        CHK(clSetKernelArg(k,0,sizeof A,&A)); CHK(clSetKernelArg(k,1,sizeof O,&O));
        CHK(clSetKernelArg(k,2,sizeof steps,&steps));
        if(t1==0) t1=timeit(k,(size_t)g_cus*l,l,3);         // 1 wg/CU reference
        double d=timeit(k,(size_t)g_cus*8*l,l,3)/8.0;        // 8 wgs/CU, per round
        printf("  %5d K %10.0f %9.2fx %6.0f K\n",
               kib[i],d*1e6,t1/d,kib[i]*(t1/d));
        clReleaseKernel(k); clReleaseProgram(p);
    }
    printf("\n  speedup is how many work-groups of that size run concurrently on one\n"
           "  CU; 'implied' is size x speedup, i.e. the LDS budget being divided.\n"
           "  It stops rising once something other than LDS caps residency.\n");
    free(h); clReleaseMemObject(A); clReleaseMemObject(O);
}

// ---------------------------------------------------------------- launch / xfer
static void t_launch(void)
{
    cl_int err; cl_program p=build("__kernel void empty(void){}\n","");
    cl_kernel k=clCreateKernel(p,"empty",&err); CHK(err);
    size_t g=64,l=64;
    for(int i=0;i<20;i++) CHK(clEnqueueNDRangeKernel(g_q,k,1,0,&g,&l,0,0,0));
    CHK(clFinish(g_q));
    int N=300; double t0=now();
    for(int i=0;i<N;i++){ CHK(clEnqueueNDRangeKernel(g_q,k,1,0,&g,&l,0,0,0)); CHK(clFinish(g_q)); }
    printf("  enqueue + clFinish round-trip  %8.1f us\n",(now()-t0)/N*1e6);
    N=3000; t0=now();
    for(int i=0;i<N;i++) CHK(clEnqueueNDRangeKernel(g_q,k,1,0,&g,&l,0,0,0));
    CHK(clFinish(g_q));
    double d=(now()-t0)/N;
    printf("  pipelined (one clFinish)       %8.1f us  (%.0f kernels/s)\n",d*1e6,1.0/d);
    clReleaseKernel(k); clReleaseProgram(p);
}

static void t_xfer(void)
{
    cl_int err; size_t bytes=256ull<<20;
    void *h=aligned_alloc(4096,bytes); memset(h,1,bytes);
    cl_mem A=clCreateBuffer(g_ctx,CL_MEM_READ_WRITE,bytes,0,&err); CHK(err);
    CHK(clEnqueueWriteBuffer(g_q,A,CL_TRUE,0,bytes,h,0,0,0));
    double best=1e30;
    for(int r=0;r<3;r++){ double t0=now(); CHK(clEnqueueWriteBuffer(g_q,A,CL_TRUE,0,bytes,h,0,0,0));
        double d=now()-t0; if(d<best)best=d; }
    printf("  clEnqueueWriteBuffer H->D   %7.2f GB/s\n",(double)bytes/best/1e9);
    best=1e30;
    for(int r=0;r<3;r++){ double t0=now(); CHK(clEnqueueReadBuffer(g_q,A,CL_TRUE,0,bytes,h,0,0,0));
        double d=now()-t0; if(d<best)best=d; }
    printf("  clEnqueueReadBuffer  D->H   %7.2f GB/s\n",(double)bytes/best/1e9);
    best=1e30;
    for(int r=0;r<3;r++){ double t0=now();
        void *m=clEnqueueMapBuffer(g_q,A,CL_TRUE,CL_MAP_WRITE,0,bytes,0,0,0,&err); CHK(err);
        memcpy(m,h,bytes);
        CHK(clEnqueueUnmapMemObject(g_q,A,m,0,0,0)); CHK(clFinish(g_q));
        double d=now()-t0; if(d<best)best=d; }
    printf("  map + memcpy         H->D   %7.2f GB/s\n",(double)bytes/best/1e9);
    free(h); clReleaseMemObject(A);
}

int main(int argc,char**argv)
{
    const char *only = argc>1 ? argv[1] : "all";
    cl_platform_id plat; cl_uint n;
    CHK(clGetPlatformIDs(1,&plat,&n));
    if(clGetDeviceIDs(plat,CL_DEVICE_TYPE_GPU,1,&g_dev,&n)!=CL_SUCCESS){
        fprintf(stderr,"no GPU device - did you set RUSTICL_ENABLE=radeonsi ?\n"); return 1; }
    clGetDeviceInfo(g_dev,CL_DEVICE_MAX_COMPUTE_UNITS,sizeof g_cus,&g_cus,0);
    cl_int err;
    g_ctx=clCreateContext(0,1,&g_dev,0,0,&err); CHK(err);
    g_q=clCreateCommandQueueWithProperties(g_ctx,g_dev,(cl_queue_properties[]){0},&err); CHK(err);

    struct { const char *nm; void (*fn)(void); } tests[] = {
        {"info",t_info},{"alu",t_alu},{"bw",t_bw},{"cache",t_cache},
        {"lds",t_lds},{"atomics",t_atomics},{"lat",t_lat},{"tile",t_tile},
        {"occ",t_occ},{"launch",t_launch},{"xfer",t_xfer},
    };
    for(unsigned i=0;i<sizeof tests/sizeof*tests;i++){
        if(strcmp(only,"all") && strcmp(only,tests[i].nm)) continue;
        printf("\n=== %s ===\n",tests[i].nm);
        tests[i].fn();
    }
    return 0;
}
