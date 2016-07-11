'''
Created on Jul 10, 2016

@author: ssudholt
'''
import os

def save_prototxt(file_path, proto_object, header_comment=None):
    with open(file_path, 'w') as output_file:
        if header_comment is not None:
            output_file.write('#' + header_comment + '\n')
        output_file.write(str(proto_object))

def read_list(file_path):        
    if not os.path.exists(file_path):
        raise ValueError('File for reading list does NOT exist: ' + file_path)    
    linelist = []    
    with open(file_path) as stream:
        for line in stream:
            line = line.strip()
            if line != '':
                linelist.append(line)
            
    
    return linelist

def write_list(file_path, line_list):
    '''
    Writes a list into the given file object
    
    file_path: the file path that will be written to
    line_list: the list of strings that will be written
    '''    
    with open(file_path, 'w') as f:
        for l in line_list:
                f.write(str(l) + '\n')