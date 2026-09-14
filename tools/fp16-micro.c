// fp16-micro.c - does half precision buy anything on this GPU through rusticl?
//
// Pin the GPU clock first (busctl ... SetRange uu 2000 2000) or the governor
// ramps during the float rows and flatters the half ones.
//
//   gcc -O2 -o fp16-micro tools/fp16-micro.c -lOpenCL
//   RUSTICL_ENABLE=radeonsi ./fp16-micro
//
// Times the same dependent mad() chain in float, half, half2 and float2,
// 8 independent chains per work-item so the ALUs have something to overlap.
// gfx1013 has packed fp16 (v_pk_fma_f16: two fp16 FMAs per lane per cycle),
// so half2 should reach ~2x the float GFLOP/s if the compiler uses it.
// The ratio is the answer; absolute numbers include launch overhead.
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

static cl_device_id g_dev; static cl_context g_ctx; static cl_command_queue g_q;

static cl_program build(const char *src,const char *opts)
{
    cl_int err; cl_program p = clCreateProgramWithSource(g_ctx,1,&src,0,&err); CHK(err);
    if(clBuildProgram(p,1,&g_dev,opts,0,0)!=CL_SUCCESS){
        size_t n; clGetProgramBuildInfo(p,g_dev,CL_PROGRAM_BUILD_LOG,0,0,&n);
        char *log=malloc(n+1); clGetProgramBuildInfo(p,g_dev,CL_PROGRAM_BUILD_LOG,n,log,0);
        fprintf(stderr,"build failed:\n%s\n",log); exit(1);
    }
    return p;
}
static double timeit(cl_kernel k,size_t g,size_t l,int reps)
{
    CHK(clEnqueueNDRangeKernel(g_q,k,1,0,&g,&l,0,0,0)); CHK(clFinish(g_q));
    double best=1e30;
    for(int r=0;r<reps;r++){ double t0=now(); CHK(clEnqueueNDRangeKernel(g_q,k,1,0,&g,&l,0,0,0)); CHK(clFinish(g_q)); double d=now()-t0; if(d<best) best=d; }
    return best;
}

// T = element type, N = flops per mad per work-item-chain-step (2 * vector width)
static const char *TMPL =
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
"__kernel void k(__global T *o, float fa, float fb, int n){\n"
"  T a = (T)(fa), b = (T)(fb);\n"
"  T x0=(T)(get_global_id(0)*1e-4f), x1=x0+(T)(0.1f), x2=x0+(T)(0.2f), x3=x0+(T)(0.3f),\n"
"    x4=x0+(T)(0.4f), x5=x0+(T)(0.5f), x6=x0+(T)(0.6f), x7=x0+(T)(0.7f);\n"
"  for(int i=0;i<n;i++){\n"
"    x0=mad(x0,a,b); x1=mad(x1,a,b); x2=mad(x2,a,b); x3=mad(x3,a,b);\n"
"    x4=mad(x4,a,b); x5=mad(x5,a,b); x6=mad(x6,a,b); x7=mad(x7,a,b);\n"
"  }\n"
"  o[get_global_id(0)] = x0+x1+x2+x3+x4+x5+x6+x7;\n"
"}\n";

int main(void)
{
    cl_platform_id plat; CHK(clGetPlatformIDs(1,&plat,0));
    CHK(clGetDeviceIDs(plat,CL_DEVICE_TYPE_GPU,1,&g_dev,0));
    cl_int err; g_ctx=clCreateContext(0,1,&g_dev,0,0,&err); CHK(err);
    g_q=clCreateCommandQueueWithProperties(g_ctx,g_dev,0,&err); CHK(err);
    char ext[4096]={0}; clGetDeviceInfo(g_dev,CL_DEVICE_EXTENSIONS,sizeof ext,ext,0);
    printf("cl_khr_fp16: %s\n", strstr(ext,"cl_khr_fp16")?"yes":"NO");

    struct { const char *name,*T; int width; } c[] = {
        {"float ", "float", 1}, {"float2", "float2", 2}, {"float4", "float4", 4},
        {"half  ", "half", 1},  {"half2 ", "half2", 2},  {"half4 ", "half4", 4}, {"half8 ", "half8", 8},
    };
    size_t l=256, g=40*8*l; int n=getenv("N")?atoi(getenv("N")):20000; float a=0.999f,b=0.001f;
    cl_mem out=clCreateBuffer(g_ctx,CL_MEM_WRITE_ONLY,g*16,0,&err); CHK(err);
    printf("  %-7s %9s %10s   %s\n","type","ms","GFLOP/s","vs float");
    double base=0;
    for(unsigned i=0;i<sizeof c/sizeof*c;i++){
        char src[2048]; snprintf(src,sizeof src,"#define T %s\n%s",c[i].T,TMPL);
        cl_program p=build(src,getenv("OPTS")?getenv("OPTS"):"-cl-mad-enable -cl-fast-relaxed-math");
        cl_kernel k=clCreateKernel(p,"k",&err); CHK(err);
        CHK(clSetKernelArg(k,0,sizeof out,&out)); CHK(clSetKernelArg(k,1,sizeof a,&a));
        CHK(clSetKernelArg(k,2,sizeof b,&b)); CHK(clSetKernelArg(k,3,sizeof n,&n));
        double t=timeit(k,g,l,5);
        double flops=(double)g*n*8*2*c[i].width;
        double gf=flops/t/1e9; if(i==0) base=gf;
        printf("  %-7s %9.3f %10.1f   %.2fx\n",c[i].name,t*1e3,gf,gf/base);
        clReleaseKernel(k); clReleaseProgram(p);
    }
    return 0;
}
