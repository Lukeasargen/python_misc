import itertools
import random
import time

import numpy as np
import pandas as pd

# Kings Corner
# ace low
# alternate red and black cards in descending order

# deal cards to each player
# remaining cards in the deck go in the middle pile
# turn over 4 top cards, cross piles
# start turn by drawing a card from the middle pile
# player makes as many possible valid movies
# first player to lay all cards wins

# valid moves
# kings can be placed in the corner piles
# check if pile top card can be placed on bottom of other piles
# check if player cards can be placed on any pile
# place player cards in vacated cross piles


ranks = {
    0: "Ace",
    1: "2",
    2: "3",
    3: "4",
    4: "5",
    5: "6",
    6: "7",
    7: "8",
    8: "9",
    9: "10",
    10: "Jack",
    11: "Queen",
    12: "King"
}

suits = {
    0: "Hearts",
    1: "Spades",
    2: "Diamonds",
    3: "Clubs"
}

def filter_deck(deck, rank=[], suit=[]):
    if type(rank)==int: rank = [rank]
    if type(suit)==int: suit = [suit]
    for r,s in deck:
        if r in rank and s in suit:
            yield (r,s)
        if r in rank and len(suit)==0:
            yield (r,s)
        if len(rank)==0 and s in suit:
            yield (r,s)

def get_shuffled_deck():
    deck = list(itertools.product(ranks, suits))
    random.shuffle(deck)
    return deck

def full_game(players=4, deal=7):
    max_deal = (48)//players  # 48 cards because there nedes to be 4 cross piles
    if deal>max_deal:
        # print(f"{deal=} is greater than {max_deal=}")
        return

    # New shuffled deck
    deck = get_shuffled_deck()
    
    # The deal
    # Give out the top of the deck to each player
    player_decks = [deck[i*deal:i*deal+deal] for i in range(players)]
    # Take the top 4 cards for the cross piles list
    cross_piles = [ [deck[deal*players + i]] for i in range(4)]
    # Make the kings piles list
    kings_piles = []
    # The remaining cards go in the stockpile list
    stockpile = deck[deal*players+4:]
    # Count the total number of turns
    turn = 0

    # The play
    while True:
        # Get the current player based on the total turns
        current_player = turn%players
        current_deck = player_decks[current_player]
        # Start turn by drawing the from the stockpile
        if len(stockpile)>0:
            current_deck.append(stockpile.pop())
            current_deck.sort(reverse=True)  # sort by rank, highest on top

        # Kings=12 can be placed in the corner piles
        for king in filter_deck(current_deck, rank=12):
            kings_piles.append([king])  # Add the card in a list
            current_deck.remove(king)  # Remove the card for the player deck
        for pile in cross_piles:
            top_card = pile[0]
            if top_card[0]==12:
                kings_piles.append(pile)  # Put the whole pile in the corner
                cross_piles.remove(pile)  # Remove the pile from the cross piles
        
        # Strategy starts here
        made_move = True
        while made_move:
            made_move = False

            # For each cross pile, check if the top card can go on the bottom of other piles
            for pile in cross_piles:
                top_card = pile[0]  # top card
                for other_pile in cross_piles+kings_piles:
                    # if the other pile has no cards, skip it
                    if len(other_pile)==0: continue
                    bottom_card = other_pile[-1]  # bottom card
                    # check if the rank above the top card is the bottom card rank
                    # check that the suit is opposite color
                    moved_pile = False
                    if top_card[0]+1==bottom_card[0] and (top_card[1]%2)!=(bottom_card[1]%2):
                        other_pile.extend(pile)  # Extend the other pile with the whole cross pile
                        cross_piles.remove(pile)  # Remove the pile from the cross pile
                        moved_pile = True
                        made_move = True
                    if moved_pile:  # cant move the same pile twice
                        break
                        
            # Check if player cards can be placed on any pile
            for top_card in current_deck.copy():
                for pile in cross_piles+kings_piles:
                    # if the other pile has no cards, skip it
                    if len(other_pile)==0: continue
                    bottom_card = pile[-1]  # bottom card
                    # check if the rank above the top card is the bottom card rank
                    # check that the suit is opposite color
                    moved_card = False
                    if top_card[0]+1==bottom_card[0] and (top_card[1]%2)!=(bottom_card[1]%2):
                        pile.append(top_card)  # Append the card to the pile
                        current_deck.remove(top_card)  # Remove the card from the player deck
                        moved_card = True
                        made_move = True
                    if moved_card:  # cant move the same card twice
                        break

            # Now there could be blank cross piles, so player starts the pile
            while len(cross_piles)<4:
                if len(current_deck)==0: break
                # TODO : random card or highest rank card
                lay_card = max(current_deck)
                # lay_card = random.choice(current_deck)
                cross_piles.append([lay_card])
                current_deck.remove(lay_card)
                made_move = True

        # print(f"END OF TURN {turn}")
        # print(f"{current_deck=}")
        # print(f"{cross_piles=}")
        # print(f"{kings_piles=}")

        # Play has gone out, the game is over
        if len(current_deck)==0:
            break

        # if turn>=4:
        #     break

        turn += 1

    # print(f"WINNER {current_player} on turn {turn}")
    return turn


if __name__ == "__main__":
    min_players, max_players = 1, 10
    min_deal, max_deal = 1, 10
    total_games = 1000

    # 100000 simulations takes 25 seconds

    inits = list(itertools.product(range(min_players, max_players+1), range(min_deal, max_deal+1)))

    turns = np.zeros((len(inits), total_games+2))

    print(f"Total Simulations: {len(inits)*total_games}")

    cols = ["Players", "Deal"]

    t0 = time.time()

    for game_num in range(total_games):
        cols.append(f"Game_{game_num}")
        for init_num, (players, deal) in enumerate(inits):
            turn = full_game(players, deal)
            turns[init_num, game_num+2] = turn
            if game_num==0:
                turns[init_num, 0] = players
                turns[init_num, 1] = deal

    print(f"Total time: {time.time()-t0:.2f} seconds.")

    df = pd.DataFrame(data=turns, columns=cols)
    df.to_csv("kings_corner.csv", index=False)
