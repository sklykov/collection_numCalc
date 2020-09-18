#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attempt to create the  unified class for importing dependecies by compiling previously developed functionality
Reason behind - to make something functional to search / import modules if the containing it directory tree isn't
included in any project

@author: ssklykov
"""
# %% Imports
import os
# import sys


# %% Class implementation
class projectImport():
    """Class for holding all support methods for importing / searching."""
    projectTree = {}
    anchorFolder = ""
    containingFolder = ""
    osSpecificSeparator = ""

    def __init__(self, anchorFolder: str = ".git"):
        """
        Initialization along attempting to collect project structure.

        Parameters
        ----------
        anchorFolder : str, optional
            The anchor folder should be in the project main directory tree. The default is ".git" that is presented
            for version controlled projects in Git.
        Returns
        -------
        Sample of class, actually.

        """
        self.containingFolder = os.getcwd()
        self.osSpecificSeparator = os.sep
        iteration_depth = 5
        iteration = 1
        while (anchorFolder not in os.listdir(os.curdir)) and (iteration <= iteration_depth):
            os.chdir("..")
            iteration += 1
        if (iteration == 5) and (anchorFolder not in os.listdir(os.curdir)):
            print("Anchor folder didn't found")
            self.anchorFolder = None
        else:
            self.anchorFolder = os.path.abspath(os.curdir)

    def printAllSelfValues(self):
        print("folder containing this script:", self.containingFolder)
        print("specific folder separator:", self.osSpecificSeparator)
        print("Expected project tree containing the anchor folder:", self.anchorFolder)


# %% Testing some capabilities
if __name__ == "__main__":
    demo_instance = projectImport()
    demo_instance.printAllSelfValues()
    demo_instance2 = projectImport("Some non-exist folder")
    demo_instance2.printAllSelfValues()
