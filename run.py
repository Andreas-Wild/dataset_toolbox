from toolbox.data_converter import OMEConverter

if __name__ == '__main__':
    converter = OMEConverter('data/', 'test_output/')
    converter.convert()