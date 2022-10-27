

import copy


import numpy as np
import random
import time

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0

infinity = np.Infinity
random.seed(0)

class AI(object):

    # def __init__(self, chessboard_size, color, time_out):
    #     self.chessboard_size = chessboard_size
    #     #You are white or black
    #     self.color = color
    #     #the max time you should use, your algorithm's run time must not exceed the time limit.
    #     self.time_out = time_out
    #     # You need to add your decision to your candidate_list. The system will get the end of your candidate_list as your decision.
    #     self.candidate_list = []
    #     self.time_stamp = np.array([14,25])
    #     #diff,mobility,weighted_map,stability,verse_stable
    #     self.coefficient_list = np.array([[0,16,1.5,15,15],[0,15,1.3,25,25],[100,30,1,25,25]])

    #     self.search_method = 'alpha_beta'
    #     self.height = 4
    #     self.cnt = 0
    #     self.Vmap = np.array([[-600,50,-10,-5,-5,-10,50,-600],
    #                     [50,120,0,0,0,0,120,50],
    #                     [-10,0,-3,-2,-2,-3,0,-10],
    #                     [-5,0,-2,0,0,-2,0,-5],
    #                     [-5,0,-2,0,0,-2,0,-5],
    #                     [-10,0,-3,-2,-2,-3,0,-10],
    #                    [50,120,0,0,0,0,120,50],
    #                    [-600,50,-10,-5,-5,-10,50,-600]])
    def __init__(self, color,coefficient_list):
        #You are white or black
        self.color = color
        #the max time you should use, your algorithm's run time must not exceed the time limit.
        self.time_out = 1
        # You need to add your decision to your candidate_list. The system will get the end of your candidate_list as your decision.
        self.candidate_list = []
        self.coefficient_list = coefficient_list
        self.time_stamp = np.array([14,25])
        self.search_method = 'alpha_beta'
        self.height = 2
        self.cnt = 0
        self.win_cnt = 0
        self.Vmap = np.array([[-600,50,-10,-5,-5,-10,50,-600],
                        [50,120,0,0,0,0,120,50],
                        [-10,0,-3,-2,-2,-3,0,-10],
                        [-5,0,-2,0,0,-2,0,-5],
                        [-5,0,-2,0,0,-2,0,-5],
                        [-10,0,-3,-2,-2,-3,0,-10],
                       [50,120,0,0,0,0,120,50],
                       [-600,50,-10,-5,-5,-10,50,-600]])
    def is_terminal_state(self, chessboard):
        idx_non = np.where(chessboard == COLOR_NONE) 
        for i in range (len(idx_non[0])):
            if self.is_available(idx_non[0][i],idx_non[1][i], self.color, chessboard):
                return False
        for i in range (len(idx_non[0])):
            if self.is_available(idx_non[0][i],idx_non[1][i], -self.color, chessboard):
                return False
        return True
  
    def is_available(self, row,col, color, chessboard):
    #up
        i = 1
        if row != 0 and chessboard[row - 1][col] == -color:
            while True:
                if row - i == -1 or chessboard[row-i][col] == 0:
                    break
                if chessboard[row - i][col] == color:
                    return True
                i += 1
            
    #down
        i = 1
        if row != len(chessboard) - 1 and chessboard[row + 1][col] == -color:
            while True:
                if row + i == len(chessboard) or chessboard[row+i][col] == 0:
                    break
                if chessboard[row + i][col] == color:
                    return True
                i += 1
    #left
        i = 1
        if col != -1 and chessboard[row][col - 1] == -color:
            while True:
                if col - i == -1 or chessboard[row][col-i] == 0:
                    break
                if chessboard[row][col - i] == color:
                    return True
                i += 1
    #right
        i = 1
        if col != len(chessboard[0]) - 1 and chessboard[row][col + 1] == -color:
            while True:
                if col + i == len(chessboard[0]) or chessboard[row][col+i] == 0:
                    break
                if chessboard[row][col + i] == color:
                    return True
                i += 1
    #upleft
        i = 1
        if row != 0 and col != 0 and chessboard[row - 1][col - 1] == -color:
            while True:
                if row - i == -1 or col - i == -1 or chessboard[row-i][col-i] == 0:
                    break
                if chessboard[row - i][col - i] == color:
                    return True
                i += 1
    #upright
        i = 1
        if row != 0 and col != len(chessboard[0]) - 1 and chessboard[row - 1][
                col + 1] == -color:
            while True:
                if row - i == -1 or col + i == len(chessboard[0]) or chessboard[row-i][col+i] == 0:
                    break
                if chessboard[row - i][col + i] == color:
                    return True
                i += 1
    #downleft
        i = 1
        if row != len(chessboard) - 1 and col != 0 and chessboard[
                row + 1][col-1] == -color:
            while True:
                if row + i == len(chessboard) or col - i == -1 or chessboard[row+i][col-i] == 0:
                    break
                if chessboard[row + i][col - i] == color:
                    return True
                i += 1
    #downright
        i = 1
        if row != len(chessboard) - 1 and col != len(chessboard[0]) - 1 and chessboard[row + 1][col+1] == -color:
            while True:
                if row + i == len(chessboard) or col + i == len(chessboard[0]) or chessboard[row+i][col+i] == 0:
                    break             
                if chessboard[row + i][col + i] == color:
                    return True
                i += 1
        return False
 
    def update(self,chessboard,action,color):
       
        if action is None:
            return
        row = action[0]
        col = action[1]
        chessboard[row][col] = color
    
        #search eight directions, update all directions(may be more than one direction of the chessboard needs to be changed)
        #up down left right upleft upright downleft downright
        # 0: needs on update    1: needs to update
        l = [0,0,0,0,0,0,0,0]
        #up
        i = 1
 
        if row != 0 and chessboard[row - 1][col] == -color:

            while True:
                if row - i == -1 or chessboard[row-i][col] == 0:
                    break
                if chessboard[row - i][col] == color:
                    l[0] = 1
                    break
                i += 1
        if l[0] == 1:
            i = 1
            while True:
                if chessboard[row - i][col] == -color:
                    chessboard[row - i][col] = color
                else:
                    break
                i += 1
    #down
        i = 1
        if row != len(chessboard) - 1 and chessboard[row + 1][col] == -color:
            while True:
                if row + i == len(chessboard) or chessboard[row+i][col] == 0:
                    break
                if chessboard[row + i][col] == color:
                    l[1] = 1
                    break
                i += 1
        if l[1] == 1:
            i = 1
            while True:
                if chessboard[row + i][col] == -color:
                    chessboard[row+i][col] = color
                else:
                    break
                i+=1
    #left
        i = 1
        if col != -1 and chessboard[row][col - 1] == -color:
            while True:
                if col - i == -1 or chessboard[row][col-i] == 0:
                    break
                if chessboard[row][col - i] == color:
                    l[2] = 1
                    break        
                i += 1
        if l[2] == 1:
            i = 1
            while True:
                if chessboard[row][col-i] == -color:
                    chessboard[row][col-i] = color
                else:
                    break
                i += 1    
    #right
        i = 1
        if col != len(chessboard[0]) - 1 and chessboard[row][col + 1] == -color:
            while True:
                if col + i == len(chessboard[0]) or chessboard[row][col+i] == 0:
                    break
                if chessboard[row][col + i] == color:
                    l[3] = 1
                    break
                i += 1
        if l[3] == 1:
            i = 1
            while True:
                if chessboard[row][col+i] == -color:
                    chessboard[row][col+i] = color
                else:
                    break
                i += 1
    #upleft
        i = 1
        if row != 0 and col != 0 and chessboard[row - 1][col - 1] == -color:
            while True:
                if row - i == -1 or col - i == -1 or chessboard[row-i][col-i] == 0:
                    break
                if chessboard[row - i][col - i] == color:
                    l[4] = 1
                    break
                i += 1
        if l[4] == 1:
            i = 1
            while True:
                if chessboard[row - i][col-i] == -color:
                    chessboard[row - i][col-i] = color
                else:
                    break
                i += 1
    #upright
        i = 1
        if row != 0 and col != len(chessboard[0]) - 1 and chessboard[row - 1][
                col + 1] == -color:
            while True:
                if row - i == -1 or col + i == len(chessboard[0]) or chessboard[row-i][col+i] == 0:
                    break
                if chessboard[row - i][col + i] == color:
                    l[5] = 1
                    break
                i += 1
        if l[5] == 1:
            i = 1
            while True:
                if chessboard[row - i][col + i] == -color:
                    chessboard[row - i][col + i] = color
                else:
                    break
                i += 1
    #downleft
        i = 1
        if row != len(chessboard) - 1 and col != 0 and chessboard[
                row + 1][col-1] == -color:
            while True:
                if row + i == len(chessboard) or col - i == -1 or chessboard[row+i][col-i] == 0:
                    break
                if chessboard[row + i][col - i] == color:
                    l[6] = 1
                    break
                i += 1
        if l[6] == 1:
            i = 1
            while True:
                if chessboard[row + i][col - i] == -color:
                    chessboard[row + i][col - i] = color
                else:
                    break
                i += 1
    #downright
        i = 1
        if row != len(chessboard) - 1 and col != len(chessboard[0]) - 1 and chessboard[row + 1][col+1] == -color:
            while True:
                if row + i == len(chessboard) or col + i == len(chessboard[0]) or chessboard[row+i][col+i] == 0:
                    break             
                if chessboard[row + i][col + i] == color:
                    l[7] = 1
                    break
                i += 1
        if l[7] == 1:
            i = 1
            while True:
                if chessboard[row + i][col + i] == -color:
                    chessboard[row + i][col + i] = color
                else:
                    break
                i += 1
        return chessboard
    
    def evaluate_action(self,chessboard,action,color):
        board = self.update(chessboard,action,color)
        val = 0
        val_diff = self.evaluate_0_difference(board,color)
        # val_mobility = self.evaluate_1_mobility(board,color)
        val_weighted_chessboard = self.evaluate_2_weighted_chessboard(board,color)
        if self.cnt <= self.time_stamp[1]:
            val = 5*val_diff +   val_weighted_chessboard
        else:
            val = 15*(self.cnt-self.time_stamp[1] + 1)*val_diff + val_weighted_chessboard
        return val
    def evaluate(self,chessboard,color):
        val = 0
        val_diff = self.evaluate_0_difference(chessboard,color)
        val_mobility = self.evaluate_1_mobility(chessboard,color)
        val_weighted_chessboard = self.evaluate_2_weighted_chessboard(chessboard,color)
        val_stability,val_verseStable = self.evaluate_3_stability(chessboard,color)
        val_list = np.array([val_diff,val_mobility,val_weighted_chessboard,val_stability,val_verseStable])
        if self.cnt <= self.time_stamp[0]:
            return np.sum(self.coefficient_list[0] * val_list)
        elif self.cnt <= self.time_stamp[1]:
            return np.sum(self.coefficient_list[1] * val_list)
        else: 
            return np.sum(self.coefficient_list[2] * val_list)
    def evaluate_0_difference(self,chessboard,color):
        res = np.sum(chessboard == COLOR_WHITE) - np.sum(chessboard == COLOR_BLACK)
        if color == COLOR_BLACK:
            return res 
        else:
            return -res  
    def evaluate_1_mobility(self,chessboard,color):
 
        #the lower mobility the oppenent have, the higher the score
        opponent_mobility = len(self.get_available_list(-color,chessboard))
        my_mobility = len(self.get_available_list(color,chessboard))
        chessboard_copy = copy.deepcopy(chessboard)
        action = self.search('greedy',self.get_available_list(-color,chessboard_copy),chessboard_copy)
        if action is None:
            potential_mobility = my_mobility
        else:
            board = self.update(chessboard,action,-color)
            potential_mobility = len(self.get_available_list(color,board))
        return 0.3*my_mobility-opponent_mobility + 0.2* potential_mobility
       
    def evaluate_2_weighted_chessboard(self,chessboard,color):
        res = 0
  
        res = np.sum(self.Vmap*chessboard)*color
        return res

    def evaluate_3_stability(self,chessboard,color):
        #more chess on one edge, more bad the situation
        #rank four edges, count stable edges
        res = 0
        stable = [0,0]
        cind1 = [0,0,7,7]
        cind2 = [0,7,7,0]
        inc1 = [0,1,0,-1]
        inc2 = [1,0,-1,0]
        stop = [0,0,0,0]
        for i in range(4):
            if chessboard[cind1[i]][cind2[i]] == color:
                stop[i] = 1
                stable[0] += 1
                for j in range(1,7):
                    if chessboard[cind1[i]+inc1[i]*j][cind2[i]+inc2[i]*j] != color:
                        break
                    else:
                        stop[i] = j + 1
                        stable[1] += 1
        for i in range(4):
            if chessboard[cind1[i]][cind2[i]] == color:
                for j in range(1,7-stop[i-1]):
                    if chessboard[cind1[i]-inc1[i-1]*j][cind2[i]-inc2[i-1]*j] != color:
                        break
                    else:
                        stable[1] += 1
        verse_stable = 0
        if chessboard[0][0] == -color and chessboard[0][7] == -color:
            for i in range(1,6):
                if chessboard[0][i] == color:
                    verse_stable += 1
        if chessboard[0][0] == -color and chessboard[7][0] == -color:
            for i in range(1,6):
                if chessboard[i][0] == color:
                    verse_stable += 1
        if chessboard[7][0] == -color and chessboard[7][7] == -color:
            for i in range(1,6):
                if chessboard[7][i] == color:
                    verse_stable += 1
        if chessboard[0][7] == -color and chessboard[7][7] == -color:
            for i in range(1,6):
                if chessboard[i][7] == color:
                    verse_stable += 1
        if chessboard[0][0] == -color and chessboard[7][7] == -color:
            for i in range(1,6):
                if chessboard[i][i] == color:
                    verse_stable += 1
        if chessboard[0][7] == -color and chessboard[7][0] == -color:
            for i in range(1,6):
                if chessboard[i][7-i] == color:
                    verse_stable += 1    
        return -sum(stable) , verse_stable
    def search(self,search_method, available_list,chessboard):
        if len(available_list) == 0:
                return
        def random_search():
            #return a random sample in available list      
            return available_list[np.random.randint(len(available_list))]
        def greedy_search():
            q_values = np.zeros(len(available_list))           
            for i in range (len(available_list)):
                chessboard_copy = copy.deepcopy(chessboard)               
                q_values[i] = self.evaluate_action(chessboard_copy,available_list[i],self.color)
            return available_list[np.argmax(q_values)]
        def minmax_search():

            return
        def alpha_beta_search():
            def max_value(chessboard, alpha, beta,height):
                if self.cnt <= self.time_stamp[1]:
                    if  height == self.height:
                        return self.evaluate(chessboard,self.color), None
                else:
                    if self.is_terminal_state(chessboard) or self.cnt == 32:
                        return self.evaluate(chessboard,self.color), None
                v, action = -infinity, None
                
                for a in self.get_available_list( self.color, chessboard):
                    chessboard_copy = copy.deepcopy(chessboard)
                    v2, _ = min_value(self.update(chessboard_copy, a,self.color), alpha, beta,height+1)
                    # print(v2,a)
                    if v2 > v:
                        v ,action = v2, a
                    if v > beta:
                        break
                    if v > alpha:
                        alpha = v
                
                return v, action

            def min_value(chessboard, alpha, beta,height):
                # print(self.cnt)
                if self.cnt <= self.time_stamp[1]:
                    if  height == self.height:
                        return self.evaluate(chessboard,self.color), None
                else:
                    # print(self.is_terminal_state(chessboard))
                    if self.is_terminal_state(chessboard) or self.cnt == 32:
                        return self.evaluate(chessboard,self.color), None
                v, move = infinity, None
                for a in self.get_available_list( -self.color, chessboard):
                    chessboard_copy = copy.deepcopy(chessboard)
                    v2, _ = max_value(self.update(chessboard_copy, a,-self.color), alpha, beta,height+1)
                    if v2 < v:
                        v ,move = v2, a
                    if v <= alpha:
                        break
                    if v < beta:
                        beta = v
                    
                return v, move

            chessboard_copy = copy.deepcopy(chessboard)
            return max_value(chessboard_copy, -infinity, +infinity,0)[1]
        # print(search_method)
        if search_method == 'random':
            return random_search()
        if search_method == 'greedy':
            return greedy_search()
        if search_method == 'minmax':
            return minmax_search()
        if search_method == 'alpha_beta':
            return alpha_beta_search()
        return
    def get_available_list(self, color, chessboard):
        idx_non = np.where(chessboard == COLOR_NONE)
        available_list = []
        for i in range(len(idx_non[0])):
            if self.is_available(idx_non[0][i],idx_non[1][i], color, chessboard):
                available_list.append((idx_non[0][i],idx_non[1][i]))
        return available_list
    def go(self, chessboard):
        # Clear candidate_list, must do this step
        self.candidate_list.clear()
        self.cnt += 1
        available_list = self.get_available_list(self.color,chessboard)
        self.candidate_list = available_list
        pos = (self.search(self.search_method,available_list,chessboard))
        if pos is not None:
            self.candidate_list.append(pos)
        return self.candidate_list

    

def test(agent_a,agent_b):
    agent_a.cnt = 0
    agent_b.cnt = 0
    chessboard = np.array([[0, 0, 0, 0, 0, 0, 0, 0], 
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, -1, 0, 0, 0],
                        [0, 0, 0, -1, 1, 0, 0, 0], 
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0]]) 
    while True:
        # print("heeee")
        if agent_a.is_terminal_state(chessboard) :
            break
        else:
            # print("hahahahaha")
            # chessboard_copy = copy.deepcopy(chessboard)
            # print(chessboard)
            
            candidate_list = agent_a.go(chessboard)
            # print("hoho")
            if len(candidate_list) != 0:
                agent_a.update(chessboard,candidate_list[-1],agent_a.color)
                # print(chessboard)
                # print(agent_a.cnt)
        if agent_b.is_terminal_state(chessboard):
            break
        else:
            candidate_list = agent_b.go(chessboard)
            if len(candidate_list) != 0:
                agent_b.update(chessboard,candidate_list[-1],agent_b.color)
                # print(chessboard)
                # print(agent_b.cnt)
    cnt_a = np.sum(chessboard == 1)
    cnt_b = np.sum(chessboard == -1)
    # print(cnt_a,cnt_b)
    if cnt_a < cnt_b:
        agent_b.win_cnt += 1
    elif cnt_a == cnt_b:
        return 0
    else:
        agent_a.win_cnt += 1

def compete(coefficient_all):
    agent_list = []
    for i in coefficient_all:
        agent = AI(1,i)
        agent_list.append(agent)
    
    agent_list2 = []
    for i in coefficient_all:
        agent = AI(-1,i)
        agent_list2.append(agent)

    res_list = np.zeros(len(agent_list),int)
    num_cnt = 0
    for agent_a in agent_list:
        for agent_b in agent_list2:
            num_cnt += 1
            # print(num_cnt)
            test(agent_a,agent_b)
    
    for i in range(len(res_list)):
        res_list[i] += agent_list[i].win_cnt
        res_list[i] += agent_list2[i].win_cnt
    print(res_list)
    return res_list
def initialize(Population):
    coefficient_rand = np.random.rand(Population,3,5)
    coefficient_all = []
    coefficient_base = np.array([[0,16,1.5,15,15],[0,15,1.3,25,25],[100,30,1,25,25]])
    for i in range(Population):
        coefficient_all.append(coefficient_rand[i]+coefficient_base)
    return coefficient_all

def born(father_coefficient,mother_coefficient):
    child_coefficient = np.zeros((3,5))
    rand_list = np.random.rand(3)
    for i in range(len(rand_list)):
        if i < 0.3333:
            child_coefficient[i] = mother_coefficient[i]
            n1 = np.random.randint(5)
            n2 = np.random.rand(1)[0]
            if n2 > 0.8:
                child_coefficient[i][n1] = father_coefficient[i][n1]
        elif i < 0.66666:
            child_coefficient[i] = father_coefficient[i]
            n1 = np.random.randint(5)
            n2 = np.random.rand(1)[0]
            if n2 > 0.8:
                child_coefficient[i][n1] = mother_coefficient[i][n1]
        else:
            child_coefficient[i] = 0.5*(father_coefficient[i] + mother_coefficient[i])


            mutation = np.random.rand(3,5)*1-0.5
            child_coefficient += mutation
    return child_coefficient
def reproduce(elite_coefficient):
    coefficient_all = []
    for i in range(len(elite_coefficient)):
        for j in range(len(elite_coefficient)):
            if i != j:
               coefficient_all.append(born(elite_coefficient[i],elite_coefficient[j]))
        coefficient_all.append(elite_coefficient[i])
    return coefficient_all

def evolution(Population,elite,max_generation):
    print("________________generation_____________",0)
    coefficient_all = initialize(Population)
    res_list = compete(coefficient_all)
    idx = np.argsort(res_list)
    coefficient_all = np.array(coefficient_all)
    sorted_coefficient = coefficient_all[idx]
    elite_coefficient = sorted_coefficient[-elite:]
    name = "results/generation_0.txt"
    # print("len",len(elite_coefficient))
    with open(name,'ab') as f:
        for i in elite_coefficient:
            np.savetxt(f, i,delimiter=" ")
    for i in range(1,max_generation):
        print("___________________generation____________",i)
        coefficient_all = reproduce(elite_coefficient)
        res_list = compete(coefficient_all)
        idx = np.argsort(res_list)
        coefficient_all = np.array(coefficient_all)
        sorted_coefficient = coefficient_all[idx]
        elite_coefficient = sorted_coefficient[-elite:]
        name = "results/generation_" + str(i)+".txt"
        # f = open(name,'a')
        with open(name,'ab') as f:
            for i in elite_coefficient:
                np.savetxt(f, i,delimiter=" ")
if __name__ == '__main__':

    Population = 49
    elite = 7
    evolution(Population,elite,20)
    c = [[2.667443226210113316e-01 ,1.680531890176057885e+01, 2.075904244318421910e+00 ,1.576418865146350612e+01 ,1.501548590595926669e+01],
[9.044118520020477670e-01, 1.593580914214212996e+01, 2.235151037514111927e+00 ,2.594413766754942330e+01, 2.545640259733475119e+01],
[1.005308968814381956e+02 ,3.076034270791147662e+01, 1.365055235016227941e+00, 2.500050777235150079e+01, 2.539702096028404199e+01]]
    agent_b = AI(1,c)
    agent_a = AI(-1,[[0,16,1.5,15,15],[0,15,1.3,25,25],[100,30,1,25,25]])
    test(agent_a,agent_b)
    print(agent_a.win_cnt)
    print(agent_b.win_cnt)


