#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attempt to create the unified class for importing dependecies by compiling previously developed functionality.

Reason behind - to make something functional to search / import modules if the containing it directory tree isn't
included in any project (or to import modules from different places in a project)

@author: ssklykov
"""
# %% Imports
import os
# import sys


# %% Holding attributes for a folder in class (maybe, redundant)
class folderClass():
    """Collect all attributes of a folder - lists of subfolders, files. Size - yet to be implemented."""

    absolute_path = ""
    relative_path = ""
    subfolders = []
    files = []
    size = 0
    default_excluded_dirs = []
    # For testing capabilities below is narrowing of output
    default_excluded_folders = []

    def __init__(self, absolute_path: str, relative_path: str,
                 default_excluded_dirs: list = [".git", ".gitignore", "__pycache__"]):
        self.absolute_path = absolute_path
        self.relative_path = relative_path
        self.default_excluded_dirs = default_excluded_dirs
        self.subfolders = projectImport.getDirList(absolute_path, default_excluded_dirs)
        entries = os.listdir(absolute_path)
        self.files = []
        for entry in entries:
            if os.path.isfile(os.path.join(absolute_path, entry)):
                self.files.append(entry)

    def __str__(self):
        """For debugging by printing all info about folder after calling print(instance_this_class)."""
        overall_description = "Extended by OOP folder on the path: \n"
        overall_description += (self.absolute_path + " \n")
        overall_description += "Contains following subfolders: \n"
        overall_description += (str(self.subfolders) + "\n")
        overall_description += "And also contains following files: \n"
        overall_description += (str(self.files) + "\n")
        return overall_description

    def file_in_this_dir(self, fileName: str) -> bool:
        """
        Check if files is presented in this folder.

        Parameters
        ----------
        fileName : str
            Just name of a file.

        Returns
        -------
        bool
            Files in a folder or not.
        """
        for someFile in self.files:
            if someFile == fileName:
                return True
                break
        return False


# %% Main class implementation - for handling all helpful methods
class projectImport():
    """Class for holding all support methods for importing / searching for a file."""

    projectTree = {}
    anchorFolder = ""
    containingFolder = ""
    projectPath = ""
    osSpecificSeparator = ""
    listOfFiles = []  # For storing of file names in the project - possibly useful
    setOfFiles = ()
    listOfDirs = []
    default_excluded_dirs = []
    onlyFilesWithUniqueNames = True

    def __init__(self, anchorFolder: str = ".git", default_excluded_dirs: list = [".git", ".gitignore", "__pycache__"]):
        """
        Initialize along attempting to collect project structure.

        Parameters
        ----------
        anchorFolder : str, optional
            The anchor folder should be in the project main directory tree. The default is ".git" that is presented
            for version controlled projects in Git.

        Returns
        -------
        Sample of the class, actually.

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
                projectFolder = folderClass(self.projectPath, self.default_excluded_dirs)
                self.listOfDirs.append(projectFolder)
            # Again, checking that it's not an empty git project
            if len(folderTree) > 0:
                (rootpath, rootName) = os.path.split(self.projectPath)
                self.projectTree[rootName] = folderTree
                self.listOfDirs = projectImport.buildProjectDirList(self.projectPath, self.default_excluded_dirs,
                                                                    self.osSpecificSeparator)
                (self.listOfFiles, self.setOfFiles) = self.getAllFilesNames()
                if len(self.listOfFiles) != len(self.setOfFiles):
                    self.onlyFilesWithUniqueNames = False
            else:
                self.projectTree[self.projectPath] = "Empty"

    def make_file_importable(self, fileName: str) -> bool:
        """
        Search of specified file and make it importable to the calling script.

        Parameters
        ----------
        fileName : str
            Name of file with extension

        Returns
        -------
        bool
            Result of an operation

        """
        file_found = [False]*len(self.listOfDirs)  # Possible hook for returning where is two files with the same name
        pathToFolderWithFile = ""
        count = 0
        for i in range(len(self.listOfDirs)):
            if self.listOfDirs[i].file_in_this_dir(fileName):
                file_found[i] = True
                pathToFolderWithFile = self.listOfDirs[i].absolute_path
                count += 1
        if count > 1:
            print("There are more then 1 file in the entire project with such file name")
            return False
        elif count == 1:
            projectImport().includeFileToSysPath(pathToFolderWithFile)
            return True
        else:
            print("File not found")
            return False

    def getAllFilesNames(self) -> tuple:
        """
        Get all names of files in a project the entire class created for.

        Returns
        -------
        tuple
            (list of all file names, set with only unique file names)

        """
        listOfFiles = []
        setOfFiles = ()
        for folderClass in self.listOfDirs:
            for file in folderClass.files:
                listOfFiles.append(file)
        setOfFiles = set(listOfFiles)
        return (listOfFiles, setOfFiles)

    # %% Making of folders structure as a list containing instances of "folder" classes
    @staticmethod
    def buildProjectDirList(projectPath: str, excludedFolders: list, osSpecificSeparator: str) -> list:
        """
        Build of project directory structure as a list of "folder" classes.

        Each "folder" instance contains list of subfolders and files located in this folder.

        Parameters
        ----------
        projectPath : str
            Path to the project / root - defined during class initialization.
        excludedFolders : list
            List with names of folders that will excluded from folders listing / counting / searching path.

        Returns
        -------
        list
            List with all classes "folder" equal to all folders in the project.

        """
        listOfDirs = []
        rootPath = projectPath
        walkDone = False
        # First iteration not in the loop (for making clear for myself mostly):
        node = {}
        (root, tail) = os.path.split(rootPath)
        node[tail] = projectImport.getDirList(rootPath, excludedFolders)  # Representation of a node - folder
        j = 0
        rootName = tail  # For keeping the name of a root directory
        projectFolder = folderClass(rootPath, "/")
        # print(projectFolder)  # debug
        listOfDirs.append(projectFolder)
        # Using classes below for collecting all subfolders along walking by a project tree
        dirs_to_visit = projectImport.convert_subfolds_to_abs_paths(rootPath, projectFolder.subfolders, osSpecificSeparator)
        # print("dirs_to_visit: ", dirs_to_visit)  # debug

        # j - for only preventing any unexpected infinite loops
        while (not walkDone) and (j < 100000):
            if len(dirs_to_visit) > 0:
                current_path = dirs_to_visit[0]  # Get the path to a directory to visit
                # print(current_path)  # debug
                rel_path = projectImport.make_rel_path(rootName, current_path)
                # print(rel_path)
                new_dir = folderClass(current_path, rel_path)
                listOfDirs.append(new_dir)
                dirs_to_visit.pop(0)  # delete visited folder - for exclude it from the list of folders that should be visited
                if len(new_dir.subfolders) > 0:
                    new_dirs_to_visit = projectImport.convert_subfolds_to_abs_paths(current_path, new_dir.subfolders,
                                                                                    osSpecificSeparator)
                    dirs_to_visit += new_dirs_to_visit
                # print(new_dir)
            else:
                walkDone = True
            j += 1
        return listOfDirs

    # %% Useful static methods - used somewhere else in this script
    @staticmethod
    def convert_subfolds_to_abs_paths(root: str, subfolders: list, osSpecificSeparator: str) -> list:
        """
        Convert root path and names of subfolders to list with complete paths to these subfolders.

        Parameters
        ----------
        root : str
            Full root path (like C:/...).
        subfolders : list
            Names of subfolders.

        Returns
        -------
        list
            All created absolute paths to subfolders.

        """
        converted = []
        for subfolder in subfolders:
            updated_path = root + osSpecificSeparator + subfolder
            converted.append(updated_path)
        return converted

    @staticmethod
    def make_rel_path(root_name: str, current_path: str) -> str:
        """
        Make a relative to a project name path like "/Some_folder/Some_subfolder/". "/" - means a project entry.

        Parameters
        ----------
        root_name : str
            Name of a project.
        current_path : str
            Absolute path of current folder.

        Returns
        -------
        relative_path : str
            Relative to a project name path.

        """
        splittingDone = False
        relative_path = ""
        j = 0
        # j is only for preventing some bugs related to infinite looping over wrongly specified path
        while (not splittingDone) and (j < 500):
            (root, tail) = os.path.split(current_path)
            current_path = root
            if tail != root_name:
                # Making relative path with UNIX path separator
                relative_path = "".join([tail, "/", relative_path])
            else:
                splittingDone = True
            j += 1
        relative_path = "/" + relative_path
        # print(relative_path)  # debug
        return relative_path

    # %% Making a list of folders with some exclusion
    @staticmethod
    def getDirList(rootPath: str, excludingFolders: list) -> list:
        """
        Return of list of dirs down from specified path (specified path should point to some folder).

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
        """
        Include the absolute path to some folder in a system path for importing / searching.

        Parameters
        ----------
        absPathToFolder : str
            Absolute path to the requested folder.

        Returns
        -------
        None.

        """
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
        """
        Delete specific for git controlled projects folders .git and .gitignore from a project tree.

        Parameters
        ----------
        folderTree : list
            List with (sub-)folders names.

        Returns
        -------
        list
            Cleaned list of folders names.

        """
        i = 0
        while i < len(folderTree):
            if (folderTree[i] == '.git') or (folderTree[i] == ".gitignore"):
                folderTree.pop(i)
                # print("Deleted folder", delF)
            else:
                i += 1
        return folderTree

    def printAllSelfValues(self):
        """Print specified internal values for debugging / testing - enabling it for get output."""
        print("*********************************")
        # print("The folder containing this script:", self.containingFolder)
        # print("specific folder separator:", self.osSpecificSeparator)
        # print("Expected project tree containing the anchor:", self.projectPath)
        print("Collected paths to folders:\n")
        for folder in self.listOfDirs:
            print(folder.absolute_path)


# %% Testing some capabilities
if __name__ == "__main__":
    demo_instance = projectImport()
    demo_instance.printAllSelfValues()
    file = "NewtonMethod.py"
    print(demo_instance.make_file_importable(file))
    # import NewtonMethod  # demonstration that file could be imported
    # Below - should be fixed the internal errors raised cause calling the import method in calling chain
    # file2 = "SimpleHistoManipulations.py"
    # demo_instance.make_file_importable(file2)
    # import SimpleHistoManipulations
    # included = demo_instance.includeFileToSearchPath("LICENSE")
    # demo_instance2 = projectImport("Some non-exist folder")
    # demo_instance2.printAllSelfValues()

# For testing capabilities - some parameters
    # default_excluded_folders = [".git", ".gitignore", "Matricies_Manipulation", "Sorting", "Plotting",
    #                             "Experimental", "LinearSystems", "Interpolation", "FunctionRoots", "__pycache__",
    #                             "MonteCarlo"]
