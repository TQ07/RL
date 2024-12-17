class ProblemSolver:
    def __init__(self):
        self.general_mindset = {1: "take a deep breath, don’t fight it", 2: "this is something fun; think about how cool the subject is", 3: "this nervousness is out of my control; it’s no big deal, nothing serious", 4: "be optimistic"} 
        self.motivation_mindset = ("Be optimistic!", "Be happy!", "Be inspired!", "Be immersed by what you are doing!")
        self.break_options = ['read neuromancer', 'draw', 'talk to friends', 'arch linux' ,'hang out with friends']
    
    def solve(self, feeling):
        if feeling == "nervous":
            self.stop_nervousness()
        elif feeling == "no motivation":
            self.get_more_motivation()
        elif feeling == "need a break":
            self.take_break()
        elif feeling == "good":
            print("Look at you! Yay!!!!!!!!!!!!!!!!!!!")


    def get_more_motivation(self, subject):
        print('general motivation mindset: ', self.motivation_mindset)

        print('\nwhat subject do you want more motivation for? Options: deep learning, rl')
        subject = input()

        if subject == 'deep learning':
            print('you are creating something that could learn!')
            print('you are teaching a machine to be intelligent!')
            print('you are understanding how to give machines magic like abilities using math and computers')
            print('I love computers!!!!!!!!!!!!!!!!!!!!!!')
        if subject == 'rl':
            print()

    def stop_nervousness(self):
        print('mindset steps: ', self.general_mindset)

    def take_break(self):
        print('try these!', self.break_options)

def simulate():
    print("<it's all right!> \n<be nice to yourself!>")
    while True:
        print("\nHow are you feeling right now? Options: nervous, no motivation, need a break?")
        feeling = input()

        solver = ProblemSolver()
        solver.solve(feeling)

simulate()