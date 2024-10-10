---
title: "Connect4"
excerpt: "A multiplayer simulation of \"Connect Four\" by Hasbro using Python 3 socket programming."
collection: portfolio
permalink: "projects/connect4"
---

### The Objective

The board consists of a vertically mounted grid with 6 rows and 7 columns. 
Two players take turns dropping coins in each column. 
The objective of the game is for a player to drop his or her coins such that they create a row, column or diagonal of 4 coins. 

### Declaring a Winner

When a player gets four coins in a row (horizontally, vertically, or diagonally), the game logic registers the player as the winner of the game.

### The Board

The game board is represented by a two-dimensional matrix of 6 rows and 7 columns. Each empty space is represented by ‘0’, and the spaces taken by a player are represented by their respective characters.

### Dropping a Coin

In real life, the board is a vertically mounted grid. So, in order to simulate the effects of gravity, a coin when dropped into a particular column falls to the lowest unoccupied space. 
In case the column already has 6 coins, the user is penalised by the server skipping their turn. 

### To Run The Code
Clone [this repo](https://github.com/ashishkulkarnii/connect4-socket) and run ```server.py``` in a terminal to start the server. Make sure your system has [localhost enabled](https://www.techwalla.com/articles/how-to-install-a-localhost-server-on-windows). To start playing the game, run ```client1.py``` and ```client2.py``` each in separate terminals.
