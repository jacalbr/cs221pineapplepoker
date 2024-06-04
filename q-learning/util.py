import random
import math
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Card:
    def __init__(self, suit, rank):
        self.suit = suit
        self.rank = rank

    def __repr__(self):
        return f"{self.rank} of {self.suit}"

class Deck:
    suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King', 'Ace']

    def __init__(self):
        self.reset()

    def deal(self):
        return self.cards.pop()
    
    def dealN(self, N):
        out = []
        for i in range(N):
            out.append(self.cards.pop())
        return out
    
    def reset(self):
        self.cards = [Card(suit, rank) for suit in self.suits for rank in self.ranks]
        random.shuffle(self.cards)

class FiveHand:
    def __init__(self):
        self.cards = []
        self.limit = 5
        self.full = False

    def add_card(self, card):
        if len(self.cards) < self.limit:
            self.cards.append(card)
            if len(self.cards) == self.limit:
                self.full = True
            return True
        else:
            return False
    
    def is_flush(self):
        suits = [card.suit for card in self.cards]
        return len(set(suits)) == 1

    def is_straight(self):
        all_ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King', 'Ace']
        ranks = self.sorted_ranks()
        for i in range(9):
            if ranks == all_ranks[i:i+5]:
                return True
        if ranks == ['2', '3', '4', '5', 'Ace']:
            return True
        else:
            return False

    def poker_value(self):
        all_ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King', 'Ace']
        counts = {rank: 0 for rank in all_ranks}
        for card in self.cards:
            counts[card.rank] += 1
        
        if self.is_flush() and self.is_straight():
            return "Straight Flush"
        elif 4 in counts.values():
            return "Four of a Kind"
        elif 3 in counts.values() and 2 in counts.values():
            return "Full House"
        elif self.is_flush():
            return "Flush"
        elif self.is_straight():
            return "Straight"
        elif 3 in counts.values():
            return "Three of a Kind"
        elif list(counts.values()).count(2) == 2:
            return "Two Pair"
        elif 2 in counts.values():
            return "One Pair"
        else:
            return "High Card"
    
    def is_royal_flush(self):
        return self.poker_value == "Straight Flush" and sorted([card.rank for card in self.cards]) == ['10', 'Ace', 'Jack', 'King', 'Queen']
        
    def sorted_ranks(self):
        all_ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King', 'Ace']
        ranks = sorted([card.rank for card in self.cards], key=lambda x: all_ranks.index(x))
        return ranks


class ThreeHand:
    def __init__(self):
        self.cards = []
        self.limit = 3
        self.full = False

    def add_card(self, card):
        if len(self.cards) < self.limit:
            self.cards.append(card)
            if len(self.cards) == self.limit:
                self.full = True
            return True
        else:
            return False

    def poker_value(self):
        all_ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King', 'Ace']
        counts = {rank: 0 for rank in all_ranks}
        for card in self.cards:
            counts[card.rank] += 1
        
        if 3 in counts.values():
            return "Three of a Kind"
        elif 2 in counts.values():
            return "One Pair"
        else:
            return "High Card"
        

class Player:
    def __init__(self):
        self.board = [ThreeHand(), FiveHand(), FiveHand()]
        self.ctp = []
        self.score = 0
    
    def add_cards(self, cards):
        #The code below randomly adds cards to hands. This will be replaced by a strategy.
        for card in cards:
            remaining = [hand for hand in self.board if not hand.full]
            if remaining == []:
                print("Error: All hands full!")
                break
            random.choice(remaining).add_card(card)
    
    def choose_cards(self, cards, num):
        #The code below randomly chooses cards to hold. This will be replaced by a strategy.
        return random.sample(cards, num)
    
    def reset_board(self):
        self.board = [ThreeHand(), FiveHand(), FiveHand()]
            

class Game:
    def __init__(self):
        self.deck = Deck()
        self.players = [Player(), Player()]
        self.round = 1
        
    def compare(self, hand1, hand2):
        types = ['High Card',
                 'One Pair',
                 'Two Pair',
                 'Three of a Kind',
                 'Straight',
                 'Flush',
                 'Full House',
                 'Four of a Kind',
                 'Straight Flush']
        val1 = hand1.poker_value()
        val2 = hand2.poker_value()
        if types.index(val1) < types.index(val2):
            return -1
        elif types.index(val1) > types.index(val2):
            return 1
        else:
            all_ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King', 'Ace']
            ranks1 = [all_ranks.index(card.rank) for card in hand1.cards]
            ranks2 = [all_ranks.index(card.rank) for card in hand2.cards]
            if val1 == "Straight Flush" or val1 == "Straight" or val1 == "Flush" or val1 == "High Card":
                ranks1 = sorted(ranks1)
                ranks2 = sorted(ranks2)
                
            elif val1 == 'Four of a Kind' or val1 == 'Three of a Kind' or val1 == "Two Pair" or val1 == 'One Pair':
                dups1 = [rank for rank in ranks1 if ranks1.count(rank) > 1]
                regs1 = [rank for rank in ranks1 if ranks1.count(rank) == 1]
                ranks1 = sorted(regs1) + sorted(dups1)          
                
                dups2 = [rank for rank in ranks2 if ranks2.count(rank) > 1]
                regs2 = [rank for rank in ranks2 if ranks2.count(rank) == 1]
                ranks2 = sorted(regs2) + sorted(dups2)
            
            elif val1 == "Full House":
                trip1 = [rank for rank in ranks1 if ranks1.count(rank) == 3]
                dubs1 = [rank for rank in ranks1 if ranks1.count(rank) == 2]
                ranks1 = sorted(dubs1) + sorted(trip1)
                
                trip2 = [rank for rank in ranks2 if ranks2.count(rank) == 3]
                dubs2 = [rank for rank in ranks2 if ranks2.count(rank) == 2]
                ranks2 = sorted(dubs2) + sorted(trip2)
            
            
            start = -1
            end = -1 - min(len(ranks1), len(ranks2))
            for i in range(start, end, -1):
                if ranks1[i] < ranks2[i]:
                    return -1
                elif ranks1[i] > ranks2[i]:
                    return 1
            return 0
            
    def calcbonus(self, hand, pos):
        val = hand.poker_value()
        all_ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King', 'Ace']
        middle = {'High Card': 0,
                  'One Pair': 0,
                  'Two Pair': 0,
                  'Three of a Kind': 2,
                  'Straight': 4,
                  'Flush': 8,
                  'Full House': 12,
                  'Four of a Kind': 20,
                  'Straight Flush': 30
                 }
        bottom = {'High Card': 0,
                  'One Pair': 0,
                  'Two Pair': 0,
                  'Three of a Kind': 0,
                  'Straight': 2,
                  'Flush': 4,
                  'Full House': 6,
                  'Four of a Kind': 10,
                  'Straight Flush': 15
                 }
        ranks = [all_ranks.index(card.rank) for card in hand.cards]
        if pos == 0:
            if val == "One Pair":
                dup = [rank for rank in ranks if ranks.count(rank) > 1][0]
                if dup >= 4: #6 has index 4
                    return dup-3 #66: +1, 77: +2, ..., AA: +9
                else:
                    return 0
            elif val == "Three of a Kind":
                dup = ranks[0] #2 has index 0
                return dup+10 #222: +10, 333: +11, ..., AAA: +22
            else:
                return 0
        elif pos == 1:
            if hand.is_royal_flush():
                return 50
            else:
                return middle[val]
        else:
            if hand.is_royal_flush():
                return 25
            else:
                return bottom[val]
            
    def is_foul(self, board):
        if self.compare(board[0], board[1]) == 1:
            return True
        if self.compare(board[1], board[2]) == 1:
            return True
        return False

    def play_round(self):
        self.deck.reset()
        for player in self.players:
            player.reset_board()
        
        
        #First Round
        for player in self.players:
            player.add_cards(self.deck.dealN(5))
        
        #Rounds Two through Five
        for i in range(4):
            for player in self.players:
                player.add_cards(player.choose_cards(self.deck.dealN(3), 2))
        
        for player in self.players:
            for hand in player.board:
                print(hand.cards)
                print(hand.poker_value())
                
        foul0 = self.is_foul(self.players[0].board)
        foul1 = self.is_foul(self.players[1].board)
        
        if foul0 and foul1:
            margin = 0
        elif foul0:
            margin = -6
        elif foul1:
            margin = 6
        else:
            margin = 0
            for i in range(3):
                margin += self.compare(self.players[0].board[i], self.players[1].board[i])
            if margin == 3:
                margin = 6
            elif margin == -3:
                margin = -6
       
        bonus0 = 0 if foul0 else sum([self.calcbonus(self.players[0].board[i], i) for i in range(3)])
        bonus1 = 0 if foul1 else sum([self.calcbonus(self.players[1].board[i], i) for i in range(3)])

        print(margin, bonus0, bonus1)
        
        self.players[0].score += margin + bonus0 - bonus1
        self.players[1].score += -margin + bonus1 - bonus0
        
        
class Agent:
    def __init__(self):
        self.board = [ThreeHand(), FiveHand(), FiveHand()]
        self.ctp = []
        self.score = 0

    def add_card(self, loc):
        #Adds the first card in ctp to the location in loc
        card = self.ctp.pop()
        self.board[loc].add_card(card)
    
    def discard(self, ind):
        self.ctp.pop(ind)
    
    def reset_board(self):
        self.board = [ThreeHand(), FiveHand(), FiveHand()]
        



class AgentGame(Game):
    def __init__(self):
        self.deck = Deck()
        self.players = [Agent(), Player()]
        self.round = 1
        self.agent_total = 0
        self.last_agent_total = 0
        
    def prep_agent(self):
        agent = self.players[0]
        if self.round == 1:
            agent.ctp = self.deck.dealN(5)
        else:
            agent.ctp = self.deck.dealN(3)
        
    def sim_player(self):
        player = self.players[1]
        if self.round == 1:
            player.add_cards(self.deck.dealN(5))
        else:
            player.add_cards(player.choose_cards(self.deck.dealN(3), 2))
        
        if self.round == 5:
            self.get_results()
        else:
            self.round += 1
            self.prep_agent()
    
    def get_results(self):
        foul0 = self.is_foul(self.players[0].board)
        foul1 = self.is_foul(self.players[1].board)
        
        if foul0 and foul1:
            margin = 0
        elif foul0:
            margin = -6
        elif foul1:
            margin = 6
        else:
            margin = 0
            for i in range(3):
                margin += self.compare(self.players[0].board[i], self.players[1].board[i])
            if margin == 3:
                margin = 6
            elif margin == -3:
                margin = -6
       
        bonus0 = 0 if foul0 else sum([self.calcbonus(self.players[0].board[i], i) for i in range(3)])
        bonus1 = 0 if foul1 else sum([self.calcbonus(self.players[1].board[i], i) for i in range(3)])
        
        self.agent_total = margin + bonus0 - bonus1
        
        self.players[0].score += margin + bonus0 - bonus1
        self.players[1].score += -margin + bonus1 - bonus0
        
        self.reset_game()
    
    def reset_game(self):
        agent = self.players[0]
        self.deck.reset()
        for player in self.players:
            player.reset_board()
        self.round = 1
        self.last_agent_total = self.agent_total
        self.agent_total = 0
        self.prep_agent()
        return (tuple(agent.ctp), tuple(agent.board))
        
    def submit_action(self, action):
        #returns NextState, Reward, Terminated
        agent = self.players[0]
        a_type, a_value = action
        
        if a_type == "Discard": #discards the card at index a_value
            agent.discard(a_value)
        elif a_type == "Place": #places the first card in agent's ctp at the location determined by a_value
            agent.add_card(a_value)
        
        if agent.ctp == []:
            self.sim_player()
            if self.round == 1:
                return (None, self.last_agent_total, 1)
        return ((tuple(agent.ctp), tuple(agent.board)), 0, 0)
    
    def get_actions(self):
        agent = self.players[0]
        if self.round == 1:
            return [("Place", i) for i in range(3) if not agent.board[i].full]
        else:
            if len(agent.ctp) == 3:
                return [("Discard", i) for i in range(3)]
            else:
                return [("Place", i) for i in range(3) if not agent.board[i].full]



from typing import List, Tuple, Dict, Any, Union, Optional, Iterable

StateT = Any
ActionT = Any

class RLAlgorithm: #from mountaincar
    def getAction(self, state: StateT) -> ActionT: raise NotImplementedError("Override me")

    def incorporateFeedback(self, state: StateT, action: ActionT, reward: int, nextState: StateT, terminal: bool):
        raise NotImplementedError("Override me")

class TabularQLearning(RLAlgorithm): #from mountaincar
    def __init__(self, actions: List[ActionT], discount: float,
                 explorationProb: float = 0.2, initialQ: float = 0):
        '''
        - actions: the list of valid actions
        - discount: a number between 0 and 1, which determines the discount factor
        - explorationProb: the epsilon value indicating how frequently the policy returns a random action
        - intialQ: the value for intializing Q values.
        '''
        self.actions = actions
        self.discount = discount
        self.explorationProb = explorationProb
        self.Q = defaultdict(lambda: initialQ)
        self.numIters = 0

    def getAction(self, state: StateT, explore: bool = True) -> ActionT:
        if explore:
            self.numIters += 1
        explorationProb = self.explorationProb
        if self.numIters < 2e4: # explore
            explorationProb = 1.0
        elif self.numIters > 1e5: # Lower the exploration probability by a logarithmic factor.
            explorationProb = explorationProb / math.log(self.numIters - 100000 + 1)
        if not explore or random.random() > explorationProb:
            qs = [self.Q[state, action] for action in self.actions]
            maxq = max(qs)
            return self.actions[max([key for key, value in enumerate(qs) if value == maxq])]
        return random.choice(self.actions)

    # Call this function to get the step size to update the weights.
    def getStepSize(self) -> float:
        return 0.1

    def incorporateFeedback(self, state: StateT, action: ActionT, reward: float, nextState: StateT, terminal: bool) -> None:
        if terminal:
            V = 0
        else:
            V = self.Q[nextState, self.getAction(nextState, False)]
        self.Q[state, action] = (1-self.getStepSize()) * self.Q[state, action] + self.getStepSize() * (reward + self.discount * V)


class MDP: #from mountaincar
    # Return the start state.
    def startState(self): raise NotImplementedError("Override me")
    
    # Property holding the set of possible actions at each state.
    @property
    def actions(self) -> List[ActionT]: raise NotImplementedError("Override me")

    # Property holding the discount factor
    @property
    def discount(self): raise NotImplementedError("Override me")

    # property holding the maximum number of steps for running the simulation.
    @property
    def timeLimit(self) -> int: raise NotImplementedError("Override me")

    # Transitions the MDP
    def transition(self, action): raise NotImplementedError("Override me")


def simulate(mdp: MDP, rl: RLAlgorithm, numTrials=10, train=True, verbose=False, demo=False): #from mountaincar
    end_score = 0
    totalRewards = []  # The discounted rewards we get on each trial
    for trial in range(numTrials):
        state = mdp.startState()
        if demo:
            mdp.env.render()
        totalDiscount = 1
        totalReward = 0
        trialLength = 0
        for _ in range(mdp.timeLimit):
            if demo:
                time.sleep(0.05)
            action = rl.getAction(state, explore=train)
            if action is None: 
                break
            nextState, reward, terminal = mdp.transition(action)
            end_score += reward
            trialLength += 1
            if train:
                rl.incorporateFeedback(state, action, reward, nextState, terminal)
            
            totalReward += totalDiscount * reward
            totalDiscount *= mdp.discount
            state = nextState

            if terminal:
                break # We have reached a terminal state

        if verbose:
            print(("Trial %d (totalReward = %s, Length = %s)" % (trial, totalReward, trialLength)))
        totalRewards.append(totalReward)
    return totalRewards, end_score


class PipoMDP(MDP):
    def __init__(self):
        self.game = AgentGame()
    
    @property
    def timeLimit(self) -> int:
        return 100 #placeholder
    
    @property
    def discount(self):
        return 0.99 #placeholder
    
    @property
    def actions(self) -> List[ActionT]:
        return self.game.get_actions()
    
    def startState(self):
        return self.game.reset_game()
    
    def transition(self, action):
        return self.game.submit_action(action)


class DummyQLearning(RLAlgorithm):
    def __init__(self, actions: List[ActionT], discount: float,
                 explorationProb: float = 0.2, initialQ: float = 0):
        self.actions = actions

    def getAction(self, state: StateT, explore: bool = True) -> ActionT:
        return random.choice(self.actions)

    def getStepSize(self) -> float:
        return 0.1

    def incorporateFeedback(self, state: StateT, action: ActionT, reward: float, nextState: StateT, terminal: bool) -> None:
        return


class QNet(nn.Module):
    def __init__(self, inputs, hidden, outputs):
        super(QNet, self).__init__()
        hidden1, hidden2 = hidden
        self.fc1 = nn.Linear(inputs, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, outputs)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


class QWrapper():
    def __init__(self, inputs, hidden, outputs, actiontable):
        self.net = QNet(inputs, hidden, outputs)
        self.actiontable = actiontable
        
    def preprocess(self, state):
        output = []
        suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King', 'Ace']
        
        ctp, board = state
        for card in ctp:
            output.append(suits.index(card.suit)+1)
            output.append(ranks.index(card.rank)+1)
        while len(output) < 10:
            output.append(0)
        
        for hand in board:
            for card in hand.cards:
                output.append(suits.index(card.suit)+1)
                output.append(ranks.index(card.rank)+1)
        while len(output) < 36:
            output.append(0)
        return torch.tensor(output).type(torch.FloatTensor)
        
    
    def getQs(self, state, actions):
        return [self.net(self.preprocess(state))[actiontable[action]] for action in actions]
    
    def getQ(self, state, action):
        return self.net(self.preprocess(state))[actiontable[action]]



class DeepQLearning(RLAlgorithm):
    def __init__(self, actions: List[ActionT], discount: float,
                 explorationProb: float = 0.2, initialQ: float = 0):
        '''
        - actions: the list of valid actions
        - discount: a number between 0 and 1, which determines the discount factor
        - explorationProb: the epsilon value indicating how frequently the policy returns a random action
        - intialQ: the value for intializing Q values.
        '''
        self.actions = actions
        self.discount = discount
        self.explorationProb = explorationProb
        
        actiontable = {("Discard", 0): 0, 
                       ("Discard", 1): 1, 
                       ("Discard", 2): 2, 
                       ("Place", 0): 3, 
                       ("Place", 1): 4, 
                       ("Place", 2): 5}
        self.Q = QWrapper(36, (32, 32), 6, actiontable)
        self.numIters = 0
        self.optimizer = optim.SGD(self.Q.net.parameters(), lr=0.01)

    def getAction(self, state: StateT, explore: bool = True) -> ActionT:
        if explore:
            self.numIters += 1
        explorationProb = self.explorationProb
        if self.numIters < 2e4: # explore
            explorationProb = 1.0
        elif self.numIters > 1e5: # Lower the exploration probability by a logarithmic factor.
            explorationProb = explorationProb / math.log(self.numIters - 100000 + 1)
        if not explore or random.random() > explorationProb:
            qs = self.Q.getQs(state, self.actions)
            maxq = max(qs)
            return self.actions[max([key for key, value in enumerate(qs) if value == maxq])]
        return random.choice(self.actions)

    # Call this function to get the step size to update the weights.
    def getStepSize(self) -> float:
        return 0.1

    def incorporateFeedback(self, state: StateT, action: ActionT, reward: float, nextState: StateT, terminal: bool) -> None:
        if terminal:
            V = 0
        else:
            V = self.Q.getQ(nextState, self.getAction(nextState, False))
            V.detach()
        self.optimizer.zero_grad()
        curQ = self.Q.getQ(state, action)
        newQ = reward + self.discount * V
        loss = (curQ - newQ)**2
        loss.backward()
        self.optimizer.step()
