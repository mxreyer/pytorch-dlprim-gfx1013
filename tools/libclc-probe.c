// libclc-probe.c - what does *this* rusticl + libclc combination give us?
//
//   gcc -O2 -o libclc-probe tools/libclc-probe.c -lOpenCL
//   RUSTICL_ENABLE=radeonsi ./libclc-probe
//
// Answers, in one run, the questions HANDOFF.md stage 3 asked (and settled
// with this probe: warning gone with the fork, no collectives with either
// libclc, fma() hardware from Mesa 26.2 on regardless of libclc):
//
//   1. Is the "Patched Mesa libclc" present?  (rusticl prints its warning to
//      stderr on the first CL call if not - nothing else to check.)
//   2. Are the OpenCL 2.0 work-group collectives (work_group_reduce_*)
//      available?  Checks CL_DEVICE_OPENCL_C_FEATURES for
//      __opencl_c_work_group_collective_functions, then just tries to build a
//      kernel that calls work_group_reduce_add().
//   3. Is fma() a hardware instruction or libclc's __clc_sw_fma emulation?
//      Times an fma() loop against the same loop written with mad(). Hardware
//      fma is ~1x mad; the software path measured ~120x in the ocl-micro
//      loop, 15-18x in this shorter one (OPENCL-PERF.md, Finding 1).
//
// Exit status is 0 regardless; read the output.
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

static cl_device_id g_dev;
static cl_context g_ctx;
static cl_command_queue g_q;

// Build; on failure print the log and return NULL instead of exiting, because
// "does this build?" is one of the questions.
static cl_program try_build(const char *src,const char *opts)
{
    cl_int err;
    cl_program p = clCreateProgramWithSource(g_ctx,1,&src,0,&err); CHK(err);
    if(clBuildProgram(p,1,&g_dev,opts?opts:"",0,0)!=CL_SUCCESS){
        size_t n; clGetProgramBuildInfo(p,g_dev,CL_PROGRAM_BUILD_LOG,0,0,&n);
        char *log=malloc(n+1); clGetProgramBuildInfo(p,g_dev,CL_PROGRAM_BUILD_LOG,n,log,0);
        // first line of the log is enough to see *why*
        char *nl=strchr(log,'\n'); if(nl) *nl=0;
        printf("    build log: %s\n",log);
        free(log); clReleaseProgram(p); return NULL;
    }
    return p;
}

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

// ---------------------------------------------------------------- features
static void t_features(void)
{
    char ver[128]={0}, cver[128]={0};
    clGetDeviceInfo(g_dev,CL_DEVICE_VERSION,sizeof ver,ver,0);
    clGetDeviceInfo(g_dev,CL_DEVICE_OPENCL_C_VERSION,sizeof cver,cver,0);
    printf("device:   %s\n          %s\n",ver,cver);

    size_t n=0;
    clGetDeviceInfo(g_dev,CL_DEVICE_OPENCL_C_FEATURES,0,0,&n);
    cl_name_version *f=malloc(n);
    clGetDeviceInfo(g_dev,CL_DEVICE_OPENCL_C_FEATURES,n,f,0);
    int cnt=n/sizeof *f, wg=0;
    printf("OpenCL C features (%d):\n",cnt);
    for(int i=0;i<cnt;i++){
        printf("    %s\n",f[i].name);
        if(!strcmp(f[i].name,"__opencl_c_work_group_collective_functions")) wg=1;
    }
    printf("__opencl_c_work_group_collective_functions: %s\n",wg?"YES":"no");
    free(f);
}

// ---------------------------------------------------------------- work_group_reduce
static void t_reduce(void)
{
    const char *src =
        "__kernel void k(__global float *o,__global const float *i){"
        "  float v = i[get_global_id(0)];"
        "  float s = work_group_reduce_add(v);"
        "  if(get_local_id(0)==0) o[get_group_id(0)] = s;"
        "}";
    const char *stds[] = { "", "-cl-std=CL2.0", "-cl-std=CL3.0" };
    for(int s=0;s<3;s++){
        printf("work_group_reduce_add with opts \"%s\": ",stds[s]);
        cl_program p = try_build(src,stds[s]);
        if(!p){ printf("    -> does NOT build\n"); continue; }
        // it built - also check it computes the right thing
        cl_int err; float in[256], out[1]={-1};
        for(int i=0;i<256;i++) in[i]=1.0f;
        cl_mem bi=clCreateBuffer(g_ctx,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,sizeof in,in,&err); CHK(err);
        cl_mem bo=clCreateBuffer(g_ctx,CL_MEM_WRITE_ONLY,sizeof out,0,&err); CHK(err);
        cl_kernel k=clCreateKernel(p,"k",&err); CHK(err);
        clSetKernelArg(k,0,sizeof bo,&bo); clSetKernelArg(k,1,sizeof bi,&bi);
        size_t g=256,l=256;
        CHK(clEnqueueNDRangeKernel(g_q,k,1,0,&g,&l,0,0,0));
        CHK(clEnqueueReadBuffer(g_q,bo,CL_TRUE,0,sizeof out,out,0,0,0));
        printf("builds, sum of 256 ones = %g (%s)\n",out[0],out[0]==256.0f?"correct":"WRONG");
        clReleaseKernel(k); clReleaseMemObject(bi); clReleaseMemObject(bo); clReleaseProgram(p);
    }
}

// ---------------------------------------------------------------- fma vs mad
static void t_fma(void)
{
    // Same dependent chain, one with fma(), one with mad(). N iterations per
    // work-item so launch overhead is noise. The result is written so nothing
    // is dead.
    const char *tmpl =
        "__kernel void k(__global float *o,float a,float b,int n){"
        "  float x = (float)get_global_id(0) * 1e-6f;"
        "  for(int i=0;i<n;i++){ x = %s(x,a,b); x = %s(x,b,a); }"
        "  o[get_global_id(0)] = x;"
        "}";
    char src_fma[512], src_mad[512];
    snprintf(src_fma,sizeof src_fma,tmpl,"fma","fma");
    snprintf(src_mad,sizeof src_mad,tmpl,"mad","mad");

    printf("building fma() kernel...\n"); cl_program pf=try_build(src_fma,0);
    printf("building mad() kernel...\n"); cl_program pm=try_build(src_mad,0);
    if(!pf||!pm){ printf("fma/mad: build failed\n"); return; }
    cl_int err;
    size_t g=40*64*8, l=64; int n=1024;      // 40 CUs, a few waves each
    cl_mem bo=clCreateBuffer(g_ctx,CL_MEM_WRITE_ONLY,g*sizeof(float),0,&err); CHK(err);
    float a=0.999f,b=0.001f;
    double t[2]; cl_program ps[2]={pf,pm};
    for(int i=0;i<2;i++){
        cl_kernel k=clCreateKernel(ps[i],"k",&err); CHK(err);
        clSetKernelArg(k,0,sizeof bo,&bo); clSetKernelArg(k,1,sizeof a,&a);
        clSetKernelArg(k,2,sizeof b,&b);   clSetKernelArg(k,3,sizeof n,&n);
        t[i]=timeit(k,g,l,10);
        clReleaseKernel(k);
    }
    double flop = (double)g*n*2*2; // 2 ops/iter, 2 flop each
    printf("fma(): %8.3f ms  %7.1f GFLOP/s\n",t[0]*1e3,flop/t[0]/1e9);
    printf("mad(): %8.3f ms  %7.1f GFLOP/s\n",t[1]*1e3,flop/t[1]/1e9);
    printf("fma/mad time ratio: %.1fx  -> fma() is %s\n",t[0]/t[1],
           t[0]/t[1] < 3 ? "HARDWARE" : "SOFTWARE (__clc_sw_fma)");
    clReleaseMemObject(bo); clReleaseProgram(pf); clReleaseProgram(pm);
}

int main(void)
{
    cl_platform_id plat; cl_uint np=0;
    CHK(clGetPlatformIDs(1,&plat,&np));
    if(!np){ fprintf(stderr,"no OpenCL platform\n"); return 1; }
    CHK(clGetDeviceIDs(plat,CL_DEVICE_TYPE_GPU,1,&g_dev,0));
    cl_int err;
    g_ctx=clCreateContext(0,1,&g_dev,0,0,&err); CHK(err);
    g_q=clCreateCommandQueueWithProperties(g_ctx,g_dev,0,&err); CHK(err);

    puts("== features"); t_features();
    puts("\n== work_group_reduce_add"); t_reduce();
    puts("\n== fma() vs mad()"); t_fma();
    return 0;
}
