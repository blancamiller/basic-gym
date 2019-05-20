#### Install Jupyter Notebook (with or without Anaconda):

https://jupyter.org/install

Please verify that you are running python 3.6x. If you are using Jupyter 2, when you open a new jupyter notebook it will say "Python 2". This is where virtual environments might be useful.

#### Install Anaconda

This is not required if you can run a jupyter notebook with Python 3. But, if you can't and don't want to update your version of python on your machine, you can use a virtual environment. For this you will need to install Anaconda.

https://docs.anaconda.com/anaconda/install/


#### Basic commands for virtual environments with Anaconda

Create a new environment with all anaconda packages:

```
conda create --name <your_envs_name> 
```

List all virtual environments:

```
conda info -e
```

Activate Environment

```
conda activate <your_envs_name>
```

Deactivate Environment

```
conda deactivate
```
