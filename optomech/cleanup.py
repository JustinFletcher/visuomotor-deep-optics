

# Delete the contents of tmp and runs
def cleanup():
    import os
    import shutil

    # Remove the contents of the tmp directory
    for filename in os.listdir('./deep_optics_gym/tmp'):
        file_path = os.path.join('tmp', filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    # Remove the contents of the runs directory
    for filename in os.listdir('./deep_optics_gym/runs'):
        file_path = os.path.join('runs', filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

# argparse cli
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Delete the contents of the tmp and runs directories.')
    args = parser.parse_args()
    cleanup()
    print('Contents of tmp and runs directories deleted.')