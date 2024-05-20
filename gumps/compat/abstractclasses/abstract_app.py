# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

import abc as interface


class AbstractApp(interface.ABC):
    """
    The interface for any application, or app. Any concrete app must inherit from this.
    The app defines a study that can be run, it executes it and presents its results, usually in a visual manner.
    It can be configured and it owns the results of the study, regardless of the success/failure of the execution.  
    It follows a Bridge and Composite design pattern.
    This corresponds to the Apogee sub-system in Cosmos.  

    FUTURE: going forward, we'd like apps to be instantiated from a simple text-based spec,
    which gets interpreted, if necessary defining a little domain-specific language.
    """

    @interface.abstractclassmethod
    def execute(self, study):
        """
        Does whatever the app needs to do.
        Returns:   
            None
        """
        pass


    @interface.abstractclassmethod
    def _specify(self, study):
        """
        Specify the app. It builds the study data
        In the future, it will also configure the runtime environment,
        if that's not put in a separate method  
        Return:  
            None
        """
        pass

    @interface.abstractclassmethod
    def _run(self):
        """
        Build the study. Execute it. Retrieve the results
        Return:  
            None
        """
        pass

    @interface.abstractclassmethod
    def _visualize(self):
        """
        Present the solution, in a visual fashion (tables or plots)
        Return:  
            None
        """
        pass

    @interface.abstractclassmethod
    def set_app_data(self, initial_state, inner_steps, outer_steps):
        pass

    @interface.abstractclassmethod
    def get_results(self):
        pass
