import random
import copy
import math

class Card:
    def __init__(self, suit, rank):
        self.suit = suit
        self.rank = rank

    def tuple_repr(self):
        return (self.suit,self.rank)

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
    
    def copy(self):
        newcopy = FiveHand()
        newcopy.cards = self.cards.copy()
        newcopy.limit = 5
        newcopy.full = self.full
        return newcopy

    def tuple_repr(self):
        hand = []
        for card in self.cards:
            hand.append(card.tuple_repr())
        return tuple(hand)

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
        ranks = sorted([card.rank for card in self.cards], key=lambda x: all_ranks.index(x))
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
    
    def points(self, location):
        all_ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King', 'Ace']
        counts = {rank: 0 for rank in all_ranks}
        for card in self.cards:
            counts[card.rank] += 1

        score = 0

        if self.is_flush() and self.is_straight() and counts["10"] == 1 and counts["Jack"] == 1 and counts["Queen"] == 1 and counts["King"] == 1 and counts["Ace"] == 1:
            score = 25
        elif self.is_flush() and self.is_straight():
            score = 15
        elif 4 in counts.values():
            score = 10
        elif 3 in counts.values() and 2 in counts.values():
            score = 6
        elif self.is_flush():
            score = 4
        elif self.is_straight():
            score = 2
        if location == "m":
            score *= 2

        elif 3 in counts.values() and location == "m":
            score = 2

        return score
    
    def hand_compare(self, hand):
        return None


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

    def copy(self):
        newcopy = ThreeHand()
        newcopy.cards = self.cards.copy()
        newcopy.limit = 3
        newcopy.full = self.full
        return newcopy

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
        
    def points(self):
        scores = {3:{'2':10, '3':11, '4':12, '5':13, '6':14, '7':15, '8':16, '9':17, '10':18, 'Jack':19, 'Queen':20, 'King':21, 'Ace':22},2:{'2':0, '3':0, '4':0, '5':0, '6':1, '7':2, '8':3, '9':4, '10':5, 'Jack':6, 'Queen':7, 'King':8, 'Ace':9}}
        all_ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King', 'Ace']
        counts = {rank: 0 for rank in all_ranks}
        for card in self.cards:
            counts[card.rank] += 1
        
        rank = max(counts.items(), key=lambda item: item[1])[0]
        if 3 in counts.values():
            return scores[3][rank]
        elif 2 in counts.values():
            return scores[2][rank]
        else:
            return 0
    
    def tuple_repr(self):
        hand = []
        for card in self.cards:
            hand.append(card.tuple_repr())
        return tuple(hand)

    def hand_compare(self, hand):
        return None
    
class CardsLeft:
    def __init__(self):
        self.reset()
    
    def card_played(self, card):
        if (card.suit, card.rank) in self.cards_remaining:
            self.cards_remaining.remove((card.suit, card.rank))
    
    def reset(self):
        self.cards_remaining = []
        suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King', 'Ace']

        for suit in suits:
            for rank in ranks:
                self.cards_remaining.append((suit,rank))



class RandomAlgo:
    def __init__(self):
        self.board = [ThreeHand(), FiveHand(), FiveHand()]
        self.score = 0
        self.options = []

    def add_cards(self, cardsleft, cards):
        #The code below randomly adds cards to hands. This will be replaced by a strategy.
        for card in cards:
            remaining = [hand for hand in self.board if not hand.full]
            if remaining == []:
                print("Error: All hands full!")
                break
            random.choice(remaining).add_card(card)
            cardsleft.card_played(card)
    
    def choose_cards(self, cards, num):
        #The code below randomly chooses cards to hold. This will be replaced by a strategy.
        return random.sample(cards, num)
    
    def reset_board(self):
        self.board = [ThreeHand(), FiveHand(), FiveHand()]
 
class PolicyGradient:
    def __init__(self):
        self.board = [ThreeHand(), FiveHand(), FiveHand()]
        self.score = 0
        self.options = {}
        self.cards_this_round = 0
        self.choices_made = {}
    
    def count_cards(self):
        return len(self.board[0].cards) + len(self.board[1].cards) + len(self.board[2].cards)

    def generate_options(self, cards):
        prev = [self.board]
        next = []
        for card in cards:
            for board in prev:
                for i in range(len(board)):
                    if not board[i].full:
                        option = [board[0].copy(),board[1].copy(),board[2].copy()]
                        option[i].add_card(card)
                        next.append(option)
                    next.append(board)
            prev = next
            next = []
        to_ret = []
        for option in prev:
            if ((len(option[0].cards) + len(option[1].cards) + len(option[2].cards)) == self.count_cards() + self.cards_this_round):
                to_ret.append(option)
        return to_ret

    def board_tuple_repr(self):
        to_ret = []
        for hand in self.board:
            to_ret.append(hand.tuple_repr())
        return tuple(to_ret)
        
    def cards_tuple_repr(self, cards):
        to_ret = []
        for card in cards:
            to_ret.append(card.tuple_repr())
        return tuple(to_ret)

    def softmax(self,options):
        e_x = [math.exp(i) for i in options]
        return [z / sum(e_x) for z in e_x]

    def choose_action(self, options):
        probs = self.softmax(list(options.values()))
        return random.choices(list(options.keys()), weights=probs, k=1)[0]

    def add_cards(self, cardsleft, cards):
        for card in cards:
            cardsleft.card_played(card)
        
        
        state = (self.board_tuple_repr(), self.cards_tuple_repr(cards), tuple(cardsleft.cards_remaining))

        if state not in self.options:
            options = self.generate_options(cards)

            state_options = {}
            for option in options:
                state_options[tuple(option)] = 0
            self.options[state] = state_options
        else:
            print("LEARN!!!!")
        self.board = self.choose_action(state_options)
        self.choices_made[state] = tuple(self.board)

    def choose_cards(self, cards, num = 5):
        self.cards_this_round = num
        return cards
    
    def update_thetas(self,score):
        for state in self.choices_made.keys():
            choice_made = self.choices_made[state]
            probs = self.softmax(self.options[state].values())
            i = 0
            for option in self.options[state].keys():
                if option == choice_made:
                    self.options[state][option] += score * (1 - probs[i])
                else:
                    self.options[state][option]  -= score * probs[i]
                i += 1

    def reset_board(self):
        self.board = [ThreeHand(), FiveHand(), FiveHand()]   



class PolicyGradientWithoutCardsRemaining:
    def __init__(self):
        self.board = [ThreeHand(), FiveHand(), FiveHand()]
        self.score = 0
        self.options = {}
        self.cards_this_round = 0
        self.choices_made = {}
    
    def count_cards(self):
        return len(self.board[0].cards) + len(self.board[1].cards) + len(self.board[2].cards)

    def generate_options(self, cards):
        prev = [self.board]
        next = []
        for card in cards:
            for board in prev:
                for i in range(len(board)):
                    if not board[i].full:
                        option = [board[0].copy(),board[1].copy(),board[2].copy()]
                        option[i].add_card(card)
                        next.append(option)
                    next.append(board)
            prev = next
            next = []
        to_ret = []
        for option in prev:
            if ((len(option[0].cards) + len(option[1].cards) + len(option[2].cards)) == self.count_cards() + self.cards_this_round):
                to_ret.append(option)
        return to_ret

    def board_tuple_repr(self):
        to_ret = []
        for hand in self.board:
            to_ret.append(hand.tuple_repr())
        return tuple(to_ret)
        
    def cards_tuple_repr(self, cards):
        to_ret = []
        for card in cards:
            to_ret.append(card.tuple_repr())
        return tuple(to_ret)

    def softmax(self,options):
        e_x = [math.exp(i) for i in options]
        return [z / sum(e_x) for z in e_x]

    def choose_action(self, options):
        probs = self.softmax(list(options.values()))
        return random.choices(list(options.keys()), weights=probs, k=1)[0]

    def add_cards(self, cardsleft, cards):
        for card in cards:
            cardsleft.card_played(card)
        
        
        state = (self.board_tuple_repr(), self.cards_tuple_repr(cards))

        if state not in self.options:
            options = self.generate_options(cards)
            state_options = {}
            for option in options:
                state_options[tuple(option)] = 0
            self.options[state] = state_options
        else:
            print("LEARN!!!!")
        self.board = self.choose_action(self.options[state])
        self.choices_made[state] = tuple(self.board)

    def choose_cards(self, cards, num = 5):
        self.cards_this_round = num
        return cards
    
    def update_thetas(self):
        score = self.board[0].points() + self.board[1].points("m") + self.board[2].points("b")
        for state in self.choices_made.keys():
            choice_made = self.choices_made[state]
            probs = self.softmax(self.options[state].values())
            index = list(self.options[state].keys()).index(choice_made)
            self.options[state][choice_made] += score * (1 - probs[index])
        return score

    def reset_board(self):
        self.board = [ThreeHand(), FiveHand(), FiveHand()]   


class Game:
    def __init__(self):
        self.deck = Deck()
        self.players = [PolicyGradientWithoutCardsRemaining(), RandomAlgo()]
        self.cardsleft = CardsLeft()

    def play_round(self):
        score = 0
        for i in range(1000000):
            self.deck.reset()
            self.cardsleft.reset()

            for player in self.players:
                player.reset_board()
            
            #First Round
            for player in self.players:
                player.add_cards(self.cardsleft, player.choose_cards(self.deck.dealN(5),5))
            
            #Rounds Two through Five
            for j in range(4):
                for player in self.players:
                    player.add_cards(self.cardsleft, player.choose_cards(self.deck.dealN(3), 2))
            
            score += self.players[0].update_thetas()
            self.players[0].choices_made = {}
            if i % 10000 == 0:
                print(i, score)
        
        for player in self.players:
            for hand in player.board:
                print(hand.cards)
                print(hand.poker_value())

        print(score)


# Example of running the game
game = Game()
game.play_round()
