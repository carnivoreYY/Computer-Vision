There are 3 files:
functions.py: storing related functions such as XYZ2Luv which will be used in another two files
program1.py: first program
program2.py: second program

how to run:
$ cd project1/
$ python program1.py 0.2 0.1 0.8 0.5 input.jpg out.png
$ python program2.py 0.2 0.1 0.8 0.5 input.jpg out.png

Explanation:
I made several decisions on my program:
1. After the conversion of RGB to Non-linear-RGB, it should be limited to (0, 1). Before the conversion of Non-linear-RGB to RGB, it should be limited to (0, 1)
2. In the function of XYZ2Luv: we handle d == 0 case. In the function of Luv2XYZ: we handle v_prime == 0 case
3. In program2.py: we handle Lmax == Lmin (step == 0) case

Description:
Normally, the results after applying my picture will be darker than original, which means that the lightness will be decrease. If I choose a very small size of window, the picture may look "bad".
