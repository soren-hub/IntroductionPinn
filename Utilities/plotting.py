import matplotlib.pyplot as plt
from PIL import Image
import os
from IPython import display
from IPython.display import Image

def display_gif(path):
    with open(path,'rb') as f:
        display(Image(data=f.read(), format='png'))
        

def mkdir(dir=None,name=None):
    '''
    Create a directory in the specified directory.
    
    :param dir: the directory to create the new directory in
    :param name: The name of the directory to be created
    '''
    try: 
        path = os.path.join("./",name) if dir==None else os.path.join(dir,name)
        os.mkdir(path)
        print("Directory '% s' created" % name)
    except FileExistsError:
        print("Directory '% s' was already created" % name)

def plot_nn_result(t,u,t_data,u_data,u_pred,epoch):

    """
    @author: bmoseley
    Pretty plot neural network training results
    """
    plt.figure(figsize=(8,4))
    
    plt.plot(t,u, color="grey", linewidth=2, alpha=0.8, label="Exact solution")
    plt.plot(t,u_pred, color="tab:blue", linewidth=4, alpha=0.8, label="Neural network prediction")
    plt.scatter(t_data,u_data, s=60, color="#00B050", alpha=0.4, label='Training data')

    l = plt.legend(loc=(1.01,0.34), frameon=False, fontsize="large")
    plt.setp(l.get_texts(), color="k")
    plt.xlim(-0.05, 2.05)
    plt.ylim(-1.1, 1.1)
    plt.text(2.1,0.7,"Training step: %i"%(epoch+1),fontsize="xx-large",color="k")
    plt.axis("off")
    
def plot_pinn_result(t, u, t_data, u_data, u_pred, epoch, mu, k, lambda_1, lambda_2):
    """
    @author: bmoseley
    Pretty plot pinn discovery and solver training results
    """
    plt.figure(figsize=(8,4))
    
    plt.plot(t,u, color="grey", linewidth=2, alpha=0.8, label="Exact solution")
    plt.plot(t,u_pred, color="tab:blue", linewidth=4, alpha=0.8, label="Neural network prediction")
    plt.scatter(t_data, u_data, s=60, color="#00B050", alpha=0.4, label='Training data')
    
    l = plt.legend(loc=(1.01,0.34), frameon=False, fontsize="large")
    plt.setp(l.get_texts(), color="k")
    plt.xlim(-0.05, 2.05)
    plt.ylim(-1.1, 1.1)
    plt.text(2.065,0.9,"Training step: %i"%(epoch+1),fontsize="xx-large",color="k")
    plt.text(2.065,0.65,r"Correct PDE : $x_{tt} + %.3f x_t + %.3f x = 0$"%(mu, k),fontsize="xx-large",color="k")
    plt.text(2.065,0.4,r"Correct PDE : $x_{tt} + %.3f x_t + %.3f x = 0$"%(lambda_1,lambda_2),fontsize="xx-large",color="k")
    plt.axis("off")

   
def save_gif(outfile, files, fps=5, loop=0):
    """"
    @author: bmoseley
    Helper function for saving GIFs
    """
    imgs = [Image.open(file) for file in files]
    imgs[0].save(fp=outfile, format='GIF', append_images=imgs[1:], save_all=True, duration=int(1000/fps), loop=loop)