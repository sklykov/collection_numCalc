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
    projectPath = ""
    osSpecificSeparator = ""
    listOFAllPathsToDirs = []
    listOfFilesNames = []
    listOfDirs = []
    default_excluded_dirs = []

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
        iteration_depth = 5  # Iteration depth for operation of going in hierarchy of folders up
        iteration = 1
        # Searching and collecting absolute path to anchor folder - that on the same level of the whole project
        while (anchorFolder not in os.listdir(os.curdir)) and (iteration <= iteration_depth):
            os.chdir("..")
            iteration += 1
        if (iteration == 5) and (anchorFolder not in os.listdir(os.curdir)):
            print("Anchor folder didn't found")
            self.anchorFolder = None
            self.projectPath = self.containingFolder
        else:
            self.projectPath = os.path.abspath(os.curdir)
            self.anchorFolder = anchorFolder
            folderTree = os.listdir(self.projectPath)  # Get all files and folders in the expected project tree
            if len(folderTree) > 0:  # If project tree is non-empty
                folderTree = projectImport.deleteGitFolders(folderTree)
            else:
                self.projectTree[self.projectPath] = "Empty"
            # Again, checking that it's not an empty git project
            if len(folderTree) > 0:
                self.projectTree[self.projectPath] = folderTree
            else:
                self.projectTree[self.projectPath] = "Empty"
            if anchorFolder == ".git":
                self.default_excluded_dirs = [".git", ".gitignore"]

    # %% Attempts to make a project structure with folders as nodes of graphs
    def buildProjectDirGraph(self):
        # Second attempt to write recursive iterator building the project folder graph
        rootPath = self.projectPath
        ignored_folders = self.default_excluded_dirs
        folders_to_visit = []  # for accounting number and place of nodes visiting
        nodesVisited = []
        walkDone = False
        # First iteration not in the loop (for making clear for myself mostly):
        numberOfDirs = len(projectImport.getDirList(rootPath))  # in a root folder
        nodesVisited = [0 for i in range(numberOfDirs)]  # for calculating the number of visits in each node
        node = {}
        (root, tail) = os.path.split(rootPath)
        node[tail] = projectImport.getDirList(rootPath)  # Representation of a node - folder
        folders_to_visit = projectImport.getDirList(rootPath)
        # print(node, "- project node (minus avoided dirs)")
        # print(folders_to_visit, " - folders to visit")
        j = 0
        current_path = self.projectPath
        nodes_to_visit = folders_to_visit
        current_node = tail
        project_structure = node
        rootName = tail  # For keeping the name of a root directory
        while (not walkDone) and (j < 1):
            (updated_node, folders_to_visit) = projectImport.update_dirs_structure(current_path, nodes_to_visit)
            if len(folders_to_visit) > 0:
                # print("node should be updated")
                # The complex path in folders forming by a string like root_folder/subfold/subsubfold
                if current_node != rootName:
                    current_node = rootName + "/" + current_node
                project_structure = projectImport.update_project_structure(current_node,
                                                                           updated_node, project_structure, rootName)
            print("updated project structure:", project_structure)
            j += 1

    @staticmethod
    def update_dirs_structure(node_path: str, array_of_folders: list) -> tuple:
        i = 0
        folders_to_visit = []
        for i in range(len(array_of_folders)):
            pathToFolder = os.path.join(node_path, array_of_folders[i])
            possible_new_dirs = projectImport.getDirList(pathToFolder)
            if len(possible_new_dirs) > 0:
                new_node = {}
                new_node[array_of_folders[i]] = possible_new_dirs
                array_of_folders[i] = new_node
                for folder in possible_new_dirs:
                    folders_to_visit.append(folder)
        return (array_of_folders, folders_to_visit)

    @staticmethod
    def update_project_structure(node_name: str, updated_node: list, project_struct: dict, rootName: str) -> dict:
        found_updated = False
        if node_name == rootName:
            project_struct[rootName] = updated_node
        return project_struct

    # %% Useful methods - used somewhere else in this script
    # For testing capabilities below is narrowing of output
    default_excluded_folders = [".git", ".gitignore", "Matricies_Manipulation", "Sorting", "Plotting",
                                "Experimental", "LinearSystems", "Interpolation", "FunctionRoots", "__pycache__",
                                "MonteCarlo"]

    @staticmethod
    def getDirList(rootPath: str, excludingFolders: list = default_excluded_folders) -> list:
        """
        Returns of list of dirs down from specified path (specified path should point to some folder).

        Parameters
        ----------
        rootPath : str
            Path to the root directory
        excludingFolders : list, optional
            List with names of folders that must be excluded from returning list.
            The default is [".git, .gitignore"].

        Returns
        -------
        list
            List with folder names that the root directory contains.

        """
        if os.path.exists(rootPath):
            listOfDirs = os.listdir(rootPath)
            i = 0
            while i < len(listOfDirs):
                pathToObj = os.path.join(rootPath, listOfDirs[i])
                if (not os.path.isdir(pathToObj)) or (listOfDirs[i] in excludingFolders):
                    listOfDirs.pop(i)
                else:
                    i += 1
            return listOfDirs
        else:
            return []

    @staticmethod
    def includeFileToSysPath(absPathToFolder: str):
        import sys
        # print("sys.path before: ", sys.path)
        if (sys.path.count(absPathToFolder) == 0):
            sys.path.append(absPathToFolder)
        elif (sys.path.count(absPathToFolder) > 1):
            for i in (2, sys.path.count(absPathToFolder)):
                sys.path.remove(absPathToFolder)
        # print("sys.path after: ", sys.path)

    @staticmethod
    def deleteGitFolders(folderTree: list) -> list:
        """Deleting specific for git controlled projects folders .git and .gitignore from a project tree"""
        i = 0
        while i < len(folderTree):
            if (folderTree[i] == '.git') or (folderTree[i] == ".gitignore"):
                folderTree.pop(i)
                # print("Deleted folder", delF)
            else:
                i += 1
        return folderTree

    def printAllSelfValues(self):
        """
        For debugging / testing - enabling it for get output
        """
        print("*********************************")
        # print("The folder containing this script:", self.containingFolder)
        # print("specific folder separator:", self.osSpecificSeparator)
        # print("Expected project tree containing the anchor:", self.projectPath)
        print("Collected paths to folders: ", self.listOFAllPathsToDirs)


# %% Testing some capabilities
if __name__ == "__main__":
    demo_instance = projectImport()
    # demo_instance.recursiveBuildDirGraph()
    # demo_instance.makeDirectoryGraph()
    # demo_instance.printAllSelfValues()
    demo_instance.buildProjectDirGraph()
    # included = demo_instance.includeFileToSearchPath("LICENSE")
    # demo_instance2 = projectImport("Some non-exist folder")
    # demo_instance2.printAllSelfValues()
