from __future__ import print_function, division
from warnings import warn
from ..node import Node
from ..utils import index_of_column_name

class Clip(Node):

    # Not very well specified.  Really want to specify that 
    # we need 'lower_limit' and 'upper_limit' to be specified in
    # each measurement...
    requirements = {'device': {'measurements': 'ANY VALUE'}}
    postconditions =  {'preprocessing_applied': {'clip': {}}}
    name = 'clip'

    def process(self):
        self.check_requirements()
        metadata = self.upstream.get_metadata()
        measurements = metadata['device']['measurements']
        for chunk in self.upstream.process():
            for measurement in chunk:
                lower, upper = _find_limits(measurement, measurements)
                if lower is not None and upper is not None:
                    icol = index_of_column_name(chunk, measurement)
                    chunk.iloc[:,icol] = chunk.iloc[:,icol].clip(lower, upper)

            yield chunk

def _find_limits(measurement, measurements):
    """
    Returns
    -------
    lower, upper : numbers
    """
    for m in measurements:
        if ((m['physical_quantity'], m['type']) == measurement):
            return m.get('lower_limit'), m.get('upper_limit')

    warn('No measurement limits for {}.'.format(measurement), RuntimeWarning)
    return None, None