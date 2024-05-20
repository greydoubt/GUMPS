# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

from gumps.compat.abstractclasses import AbstractApp
from gumps.compat.common import VariableMetaData, StudyMetaData
from gumps.compat.studies import Study

class App(AbstractApp):
    """
    The interface for any application, or app. Any concrete app must inherit from this.
    The app defines a study that can be run, it executes it and presents its results, usually in a visual manner.
    It can be configured and it owns the results of the study, regardless of the success/failure of the execution.  
    It follows a Bridge and Composite design pattern.
    This corresponds to the Apogee sub-system in Cosmos.  

    FUTURE: going forward, we'd like apps to be instantiated from a simple text-based spec,
    which gets interpreted, if necessary defining a little domain-specific language.

    Attributes

    -----------
    _app_data : tuple  
        Tuple of lists of variables, where each list is the corresponding variables for each nested study.

    """

    def __init__(self, interactive = False):
        self._app_config = {"interactive": interactive}
        # the setup token data needed to run the app
        self._app_data = None
        # the (still unstructured) results of running the app
        self._app_results = None
        # the (hacky) points that will be used for regression testing
        self._app_regression_points = None

    def set_app_data(self):
        '''Function responsible for setting application data.

        When creating a subclass from app, this should be the only function that needs to be
        overwritten. It is responsible for creating the application data that will be used
        in the study. 

        Parameters
        ----------
        None

        Returns
        -------
        None
        Should set the attribute _app_data, which should take the form of a tuple of lists of variable where the index of the tuple relates to the depth of the study being set.
        '''
        pass

    def execute(self, study):
        '''Checks that all variables needed by the study have been set and then runs the study.  

        In the case all variables have not yet been set, study will print off the key of each missing variable.

        Parameters
        ----------
        study : Study - The study to be to be executed

        Returns
        -------
        None

        FUTURE
        ------

        * Instead of printing out missing variables, raise Error.
        '''
        self._specify(study)
        self._run()
        if self._app_config["interactive"]:
            self._visualize()

    def _run(self):
        """
        Default run method. Assumes a self.study has been defined.
        """
        self.study.initialize()
        try:
            self.study.run()
        except Exception as e:
            self.study.deinitialize()
            raise e
        self.study.deinitialize()
        self._app_results = self.study.get_variables()

    def _specify(self, study):
        self.study = study(*self._app_data)
        #self._confirm_data()

    def _definedAppData(self, varKey):
        for item in self._app_data:
            if varKey in [x.symbol for x in item]:
                return True
        return False

    def _confirm_data(self):
        print("Variables not set in App Data:")
        notDefined = self._findMissing()
        for problem in notDefined:
            for missingVariable in notDefined[problem]:
                print(problem, missingVariable)
        print("_____________")

    #A Couple of open questions, how should we
    #determine the level of thte variable in app data? maybe make dictionary?
    #pass Variable registry instead of tuple?
    def _findMissing(self):
        notDefined = {}
        problemVars = self._define_problem()
        for problem, varKeys in problemVars.items():
            notDefined[problem] = []
            for varKey in varKeys:
                pass
                # if not self._definedAppData(varKey):
                #     notDefined[problem].append(varKey)
        return notDefined


    def get_results(self):
        '''Returns variable registry of the study.

        Parameters
        ----------
        None

        Returns
        -------
        variable registry
        '''
        return self._app_results

    #Can do with some cleaning, assumes structure is always 
    #study or kernel
    #TODO does not detect kernel variables
    def _getMetaData(self, study, problemMetaData):
        problemMetaData.addVarMeta(VariableMetaData(study.id, study.solver))
        nodes = study.structure.order()
        vpTargets = study.structure._get_targets()
        problemMetaData.addVpTargets(vpTargets)
        for node in nodes:
            if isinstance(node, Study): #bad!
                problemMetaData = self._getMetaData(node, problemMetaData)
            else:
                for _,childNode in node._children.items():
                    problemMetaData.addVarMeta(VariableMetaData(study.id, childNode))
        return problemMetaData

    def _define_problem(self):
        studyMeta = StudyMetaData()
        studyMeta = self._getMetaData(self.study, studyMeta)
        varsToSet = studyMeta.byProblem()
        return varsToSet

    def _visualize(self):
        pass