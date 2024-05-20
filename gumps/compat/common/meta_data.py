# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

from itertools import groupby

class VariableMetaData():
    def __init__(self, studyId, node):
        self.studyId = studyId
        self.parentId = node.id
        nodeVars = node._inspect_variables()
        self.factors = nodeVars['input']
        self.responses = nodeVars['output']
    def varRegistryKeys(self):
        return [(self.parentId, symbol) for symbol in self.factors]

class StudyMetaData():
    def __init__(self):
        self.varMeta = []
        self.vpTargets = []

    def __repr__(self):
        return repr(self.varMeta)

    def addVarMeta(self, item):
        self.varMeta.append(item)

    def addVpTargets(self, items):
        for vp in items:
            self.vpTargets.append((vp[0].id, vp[1]))

    def clean(self, node):
        return [x for x in node.varRegistryKeys() if not x in self.vpTargets]

    def byProblem(self):
        problemVars = {}
        for study, nodes in groupby(self.varMeta, lambda x: x.studyId):
            if not study in problemVars.keys():
                problemVars[study] = []
            for node in nodes:
                problemVars[study].extend(self.clean(node))
        return problemVars
