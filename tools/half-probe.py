# Which half/bfloat16 operations work in pytorch_ocl today, with error vs CPU.
# Needs torch + pytorch_ocl (build.sh makes a venv in scratch/venv). The inventory behind
# upstream/dlprimitives-winograd-fp16-inner-loop.md.
import torch, pytorch_ocl, traceback  # noqa
d='ocl:0'
def t(name, fn):
    try:
        r=fn(); torch.ocl.synchronize()
        if isinstance(r, tuple): r,ref=r; err=((r.float().cpu()-ref.float()).abs().max()/ref.float().abs().max()).item(); print(f"  {name:34s} ok   max rel err {err:.2e}  dtype {r.dtype}")
        else: print(f"  {name:34s} ok   dtype {r.dtype}")
    except Exception as e:
        print(f"  {name:34s} FAIL {str(e).splitlines()[0][:90]}")
x=torch.randn(256,256); y=torch.randn(256,256)
xh=x.to(d).half(); yh=y.to(d).half()
print("half tensors on ocl:0")
t("to(half)", lambda: (xh, x.half()))
t("add", lambda: (xh+yh, (x+y).half()))
t("mul scalar", lambda: (xh*2.5, x*2.5))
t("relu", lambda: (torch.relu(xh), torch.relu(x)))
t("exp", lambda: (torch.exp(xh*0.1), torch.exp(x*0.1)))
t("sum", lambda: (xh.sum(), x.sum()))
t("matmul", lambda: (xh@yh, x@y))
t("linear", lambda: (torch.nn.functional.linear(xh,yh), torch.nn.functional.linear(x,y)))
img=torch.randn(8,16,32,32); w=torch.randn(32,16,3,3)*0.1
t("conv2d", lambda: (torch.nn.functional.conv2d(img.to(d).half(),w.to(d).half(),padding=1), torch.nn.functional.conv2d(img,w,padding=1)))
t("batch_norm", lambda: (torch.nn.functional.batch_norm(img.to(d).half(),None,None,training=True), torch.nn.functional.batch_norm(img,None,None,training=True)))
t("max_pool2d", lambda: (torch.nn.functional.max_pool2d(img.to(d).half(),2), torch.nn.functional.max_pool2d(img,2)))
t("softmax", lambda: (torch.softmax(xh,1), torch.softmax(x,1)))
t("cross_entropy", lambda: (torch.nn.functional.cross_entropy(xh, torch.arange(256,device=d)%10), torch.nn.functional.cross_entropy(x, torch.arange(256)%10)))
print("bfloat16:")
t("bf16 add", lambda: (xh.bfloat16()+yh.bfloat16(), (x+y).bfloat16()))
t("bf16 matmul", lambda: (x.to(d).bfloat16()@y.to(d).bfloat16(), x@y))
print("autocast:")
def ac():
    with torch.autocast(device_type="ocl", dtype=torch.float16):
        return (x.to(d)@y.to(d)), x@y
t("autocast(ocl) matmul", ac)
