# README.md

This is the README.md describing how to run the code files: **intrinsics.py**, **sift.py**, **nndr.py**, **ransac.py** and **triangulation.py**

### Step 1: Virtual Environment

As always, it is best to create a virtual environment to install libraries so that they do not affect the global interpreter. 

To create a virtual environment(venv is sufficient as we are not installing heavy libraries),

```python3 -m venv <name of the virtual environment>```

```source <name of the virtual environment>/bin/activate```

Make sure the virtual environment is active before installing libraries. 

### Step 2: Installing the libraries

Before running the code, make sure the necessary libraries are installed. They are **opencv-python**, **scikit-image**, **numpy** and **viser**. Other libraries like **os**, **sys**, **time** are pre-installed in python interpreters and so, it is redundant to re-install them. It is recommended to create a virtual environment to install these libraries to prevent any conflicts with the global interpreter.

To install the libaries,

```pip3 install scikit-image numpy opencv-python viser```

**Note**: After installing the libraries, it is recommended to refresh the developer window to make sure the installs are updated to the interpreter.

### Step 3: Running the code files

I have written the script in such a way that you do not have to pass the file paths as arguments to the command line. **Note**: The image paths are given in such a way that the *assets/* folder is outside *web/*. And so, for examples, the images are referred as: ***../assets/staff_images/img1.jpeg*** and so on.

One other thing to note is that I wrote separate scripts for each step, meaning in each code file, it runs the previous steps. For example, after running intrinsics.py, when we run sift.py, it calls back intrinsics.py and the functions to perform subsequent actions. 

We can simply run the code files by:

```python3 ./code/intrinsics.py```

```python3 ./code/sift.py```

```python3 ./code/nndr.py```

```python3 ./code/ransac.py```

```python3 ./code/triangulation.py```