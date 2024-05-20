# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

import subprocess
import pkg_resources
import logging

logger = logging.getLogger(__name__)
def git(*args):
    '''Runs a git command using subprocess module and obtain its output.

    :param \*args: list of git command and option arguments in order (don't need to include "git" )
    :return: the output of running specified git command
    '''   
    return subprocess.check_output(["git"] + list(args))

def get_latest_git_commit(short_id=True):
    ''' Gets the hash ID of the latest git commit.

    **Note:** The short hash is unique within the repository. So when searching with this hash ID, make sure to search within the repository.
    
    :short_id:  If True returns the commit short hash ID, otherwise reutrns long hash ID.              
    :return:    Commit hash ID in string format
    '''
    
    if short_id:
        commit_id = git("log", "-1", "--pretty=format:'%h'").decode().strip()
    else:
        commit_id = git("log", "-1", "--pretty=format:'%H'").decode().strip()
    return str(commit_id)

def get_current_git_branch():
    ''' Gets the current git branch
    :return: Current git branch.
    '''
    git_branch = git("branch", "--show-current").decode().strip()
    return str(git_branch)


def check_uncomitted_changes():
    ''' Gets a list of uncommitted changes.
    :return: list of uncommitted changes or empty list if there are no uncommitted changes. If there are uncommitted changes, a status letter next to each 
    file indicates the status of changes. See https://git-scm.com/docs/git-diff#Documentation/git-diff.txt---diff-filterACDMRTUXB82308203 under the description 
    of the --diff-filter option on what the status letters mean.

    '''
    changes = git("diff", "--name-status").decode().strip()
    return changes.split("\n")


def version_logging():
    '''Log all the python packages and its version
    '''
    installed_packages = pkg_resources.working_set
    logger.info('Below are the python packages used')
    for pkg in installed_packages:
        logger.info(f"{pkg.key} : {pkg.version}")
