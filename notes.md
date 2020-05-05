#### May 2020. Local native install (on Windows)

Ok, finally I've made this software working on my PC (Windows 10, Nvidia 1080ti GPU). If you're going to create a similar setup, be aware that the Theano framework used by @alexjc's implementation is not supported for quite some time. Which means you shouldn't expect much support in case you step into troubles. And it is highly likely you will do so :) 

The key thing for me was this 2017 tutorial on Theano setup: https://github.com/philferriere/dlwin/blob/master/README_Jan2017.md. Thanks, @philferriere!!! If you follow it nicely, chances are you will make things work. Not going to document every single step, but here's what is in my working Anaconda env (the Python used is 3.6):

# packages in environment at D:\anaconda3\envs\theano:
# Name                    Version                   Build  Channel
blas                      1.0                         mkl    anaconda
certifi                   2020.4.5.1       py36h9f0ad1d_0    conda-forge
colorama                  0.4.3                      py_0    conda-forge
freetype                  2.10.1               ha9979f8_0    conda-forge
icc_rt                    2019.0.0             h0cc432a_1    anaconda
intel-openmp              2020.0                      166    anaconda
jpeg                      9c                hfa6e2cd_1001    conda-forge
lasagne                   0.2.dev1                 pypi_0    pypi
libblas                   3.8.0                    14_mkl    conda-forge
libcblas                  3.8.0                    14_mkl    conda-forge
liblapack                 3.8.0                    14_mkl    conda-forge
libpng                    1.6.37               hfe6a214_1    conda-forge
libpython                 2.0                      py36_0    conda-forge
libtiff                   4.1.0                h885aae3_6    conda-forge
lz4-c                     1.9.2                h62dcd97_1    conda-forge
mkl                       2019.4                      245    anaconda
numpy                     1.18.4                   pypi_0    pypi
olefile                   0.46                       py_0    conda-forge
pillow                    7.1.2            py36he4e95fe_0    conda-forge
pip                       20.1               pyh9f0ad1d_0    conda-forge
python                    3.6.10          he025d50_1009_cpython    conda-forge
python_abi                3.6                     1_cp36m    conda-forge
scipy                     1.4.1                    pypi_0    pypi
setuptools                46.1.3           py36h9f0ad1d_0    conda-forge
six                       1.14.0                   pypi_0    pypi
theano                    0.8.2                    pypi_0    pypi
tk                        8.6.10               hfa6e2cd_0    conda-forge
vc                        14.1                 h869be7e_1    conda-forge
vs2015_runtime            14.16.27012          h30e32a0_2    conda-forge
wheel                     0.34.2                     py_1    conda-forge
wincertstore              0.2                   py36_1003    conda-forge
xz                        5.2.5                h2fa13f4_0    conda-forge
zlib                      1.2.11           vc14h1cdd9ab_1  [vc14]  anaconda
zstd                      1.4.4                h9f78265_3    conda-forge

The Theano was installed using the guide cited above. I've got only two caveats:

Caveat #1: some weird issue with unicode. I have Russian Windows, so no surprise :) This is how I fixed it:

```
(theano) F:\theano-0.8.2>git diff theano/compat/__init__.py
diff --git a/theano/compat/__init__.py b/theano/compat/__init__.py
index 709dada33..50f047cb0 100644
--- a/theano/compat/__init__.py
+++ b/theano/compat/__init__.py
@@ -39,11 +39,11 @@ if PY3:
     from collections import OrderedDict, MutableMapping as DictMixin

     def decode(x):
-        return x.decode()
+        return x.decode('iso-8859-1')

     def decode_iter(itr):
         for x in itr:
-            yield x.decode()
+            yield x.decode('iso-8859-1')
 else:
     from six import get_unbound_function
     from operator import div as operator_div
```

Caveat #2: Theano is so old that it doesn't like the Nvidia driver installed on my PC (441.66). Well, let's just switch the driver check off:

```
(theano) F:\theano-0.8.2>git diff theano/sandbox/cuda/tests/test_driver.py
diff --git a/theano/sandbox/cuda/tests/test_driver.py b/theano/sandbox/cuda/tests/test_driver.py
index 8bfb05a7d..6454d039a 100644
--- a/theano/sandbox/cuda/tests/test_driver.py
+++ b/theano/sandbox/cuda/tests/test_driver.py
@@ -35,12 +35,12 @@ def test_nvidia_driver1():
         msg = '\n\t'.join(['Expected exactly one occurrence of GpuCAReduce ' +
             'but got:']+[str(app) for app in topo])
         raise AssertionError(msg)
-    if not numpy.allclose(f(), a.sum()):
-        raise Exception("The nvidia driver version installed with this OS "
-                        "does not give good results for reduction."
-                        "Installing the nvidia driver available on the same "
-                        "download page as the cuda package will fix the "
-                        "problem: http://developer.nvidia.com/cuda-downloads")
+    # if not numpy.allclose(f(), a.sum()):
+    #     raise Exception("The nvidia driver version installed with this OS "
+    #                     "does not give good results for reduction."
+    #                     "Installing the nvidia driver available on the same "
+    #                     "download page as the cuda package will fix the "
+    #                     "problem: http://developer.nvidia.com/cuda-downloads")

 def test_nvidia_driver2():
```

Other than this, no issues with installing Theano from source! Oh, you might need to install Cython, too

Lasagne was installed simply via pip:
```
pip install --upgrade --no-deps https://github.com/Lasagne/Lasagne/archive/master.zip
```

#### Running the training 

When it came to training, I realised how sensitive Theano is to THEANO_FLAGS env variable. This is the value that works for me:

```
floatX=float32,device=gpu,optimizer_including=cudnn,dnn.enabled=True,blas.ldflags=-LD:/bin/OpenBLAS-v0.2.14-Win64-int32/bin -lopenblas
```

(Yes, this line means you have to install OpenBLAS and CuDNN also, but the tutorial covers this part too)

