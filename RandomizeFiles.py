# import shutil, random, os
#
#
# def Randomize75():
#     og = os.getcwd()
#     dirpath = os.getcwd()
#     dirpath = os.path.join(dirpath, "train\\sadness")
#     destDirectory = os.path.join(og, "Training75\\sadness")
#     testDir = os.path.join(og, "test")
#     # 105 78 27
#     # 63 47 16
#     # 58 43 15
#     # 59 44 15
#     # 64 48 16
#     # 52 39 13
#     filenames = random.sample(os.listdir(dirpath), 52)
#     index = 0
#     for fname in filenames:
#         old_file = os.path.join(dirpath, fname)
#         new_file = os.path.join(dirpath, f'sadness{index}.wav')
#         index += 1
#         os.rename(old_file, new_file)
#     filenames = random.sample(os.listdir(dirpath), 39)
#     for fname in filenames:
#         srcpath = os.path.join(dirpath, fname)
#         shutil.copy(srcpath, destDirectory)
#         os.remove(os.path.join(dirpath, fname))
#     filenames = random.sample(os.listdir(dirpath), 13)
#     for fname in filenames:
#         srcpath = os.path.join(dirpath, fname)
#         shutil.copy(srcpath, testDir)
#         os.remove(os.path.join(dirpath, fname))
#
#
# Randomize75()