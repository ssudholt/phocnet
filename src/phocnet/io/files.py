'''
Created on Jul 10, 2016

@author: ssudholt
'''
def save_prototxt(file_path, proto_object, header_comment=None):
    with open(file_path, 'w') as output_file:
        if header_comment is not None:
            output_file.write('#' + header_comment + '\n')
        output_file.write(str(proto_object))