import multiprocessing
import matplotlib.pyplot as plt
import numpy as np

def main():
    if __name__ == '__main__':
        pool = multiprocessing.Pool()
        num_figs = 20
        input = zip(np.random.randint(10,1000,num_figs),
                    range(num_figs))
        pool.map(plot, input)

def plot(args):
    num, i = args
    fig = plt.figure()
    data = np.random.randn(num).cumsum()
    plt.plot(data)
    plt.title('Plot of a %i-element brownian noise sequence' % num)
    plt.pause(1)
    fig.savefig('temp_fig_%02i.png' % i)

main()