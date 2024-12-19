#import torch 
import math
import random

class Wheel:
    def __init__(self, options):
        self.options = options
    
    def spin(self):
        print(random.choice(self.options))

    def yes_or_no(self, which):
        if random.random() > 0.5:
            print('yes', which)
        else:
            print('no', which)


options = ['study c++', 'do rl', 'deep learning from scratch']
wheel = Wheel(options=options)
wheel.spin()