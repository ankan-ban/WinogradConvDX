del shaders.h

dxc /Tcs_6_2 /EInputTransform /Fh temp.txt Shaders.hlsl  -enable-16bit-types
type temp.txt >> shaders.h
del temp.txt

dxc /Tcs_6_2 /EOutputTransform /Fh temp.txt Shaders.hlsl  -enable-16bit-types
type temp.txt >> shaders.h
del temp.txt

pause