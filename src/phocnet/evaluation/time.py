'''
Created on Jul 10, 2016

@author: ssudholt
'''
def convert_secs2HHMMSS(secs):
    '''
    Takes as input a float/int representing a timing interval in seconds
    and converts it to a string in the format hh:mm:ss
    '''
    secs = int(secs)
    m, s = divmod(secs, 60)
    h, m = divmod(m, 60)
    return'%dh%02dm%02ds' % (h, m, s)