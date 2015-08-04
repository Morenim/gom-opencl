# gom-opencl

GOM-OpenCL aims to implement a fully parallelized version of the P3 algorithm by Brian W. Goldman 
and William F. Punch using the raw power of modern GPUs.

## Dependencies

The implementation relies on the OpenCL bindings provided by [rainliu](http://github.com/rainliu):
  
    go get -v -tags="cl12" github.com/rainliu/opencl/cl

To build/install the executable from the repository execute the following command:

    go get -v -tags="cl12" github.com/Morenim/gom-opencl

Only the latest OpenCL 1.2 specification is officially supported.

## Host Specifications

The executable was tested on hosts with the following specifications:

    CPU: Core i7-2600k clocked at ~3.4Ghz
    GPU: Nvidia GTX 560 Ti clocked at 882 Mhz with 384 CUDA cores and 1024 MB GDDR5 memory.
    RAM: 16 GB DDR3 clocked at 1333 Mhz
