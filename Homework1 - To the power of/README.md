# CS5330-pow
To compare C++ pow function verus simple multiplication, I have used C++ high resolution clock to time 10 iterations of the calculation respectively.

Below shows the result of the iterations.  Obviously pow takes more time, and it would significantly slow down in image processing, when consider calculating 1000 of whole images for example.

Therefore, multiplication should be used in any case.

```
Time taken pow(0.002342342342808909354,2): 1.1667e-05
Time taken (Multiplication, power = 0.002342342342808909354 * 0.002342342342808909354)2.92e-07
```
