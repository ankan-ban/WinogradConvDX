del shaders.h

dxc /Tcs_6_2 /EInputTransform_FP16 /DFP16_IO=1 /Fh temp.txt Shaders.hlsl  -enable-16bit-types
type temp.txt >> shaders.h
del temp.txt

dxc /Tcs_6_2 /EOutputTransform_FP16 /DFP16_IO=1 /Fh temp.txt Shaders.hlsl  -enable-16bit-types
type temp.txt >> shaders.h
del temp.txt

dxc /Tcs_6_2 /EInputTransform_FP32 /Fh temp.txt Shaders.hlsl  -enable-16bit-types
type temp.txt >> shaders.h
del temp.txt

dxc /Tcs_6_2 /EOutputTransform_FP32 /Fh temp.txt Shaders.hlsl  -enable-16bit-types
type temp.txt >> shaders.h
del temp.txt

pause