# PetroPy

A repository containing all the files to work toghether as a team

## What is git

Git is a Version Control System (VSC)

It's purpose it's to not lose ANY of the modification we make on a project of ours:
any action we make is saved like, and our project from git's point of view look like an album with photos on various
dates: obviously the most recent one represents the state of our project, but if we want to come back to some point or
maybe make a crazy modification and test it we can do it!

Those photos are called "commits" and we can create one with just one command.
Commits must come with a message because our collaborators or even our future self could not understand the meaning or
the reason behind certain modifications.

## Why git

First of all, having historical data of everything we do is cool. We can travel in time in any moment we'd like to,
bring our project backward or retrieve lost data with extreme ease.

Nonetheless, git is also useful to easily share your project with other developers or collaborators:
with proper committing, you can be sure that all the people participating in a project are seeing and working on the
very same files

## Git Commands

To work with the repo you can use the following commands:

`git clone <repo-url>` to save it on your computer (it will be saved in a folder named `petropy`)

`git commmit -m "<mymessage>"` creates a new save point with the text "mymessage" in it: please use meaningful messages
explaining what you did prior to that save or explaining your modifications

`git push` uploads/sends (pushes!) the changes (commits) you've made and the save points you created to the remote
server

`git pull` downloads/receives (pulls!) the changes (commits) others have made onto your local files

## Git good practices

The minimum you can do is just pull&push your project when you work on it, to be sure it's always in sync with the one
in the remote repository and the one of you collegues!

So, when you start working on the project you issue the command `git pull`: it will download the latest changes or
inform you that everything is `Already up to date.`

Then, when you work on a file, you can `git commit` from time to time, do it when it feels right to you (so when you
have made a bunch of inter-related modification)
In the end, when you are done working or editing on the project, you can `git push` to send all the commits you've made
to the remote repository!