import os
import json
import argparse
import natsort

def extract_image_numbers(files_list):
    # Extract numbers from image filenames
    return [int(f.split('.')[0]) for f in files_list if f.endswith('.png')]


def blacklist_maker(parent_directories, sub_directories, base_path, reference_directory):
    # Load existing blacklist
    blacklist_file = os.path.join(base_path, "blacklist.json")
    if os.path.exists(blacklist_file):
        with open(blacklist_file, 'r') as f:
            blacklist = json.load(f)
    else:
        blacklist = {}

    # Iterate through parent directories
    for parent_dir in parent_directories:
        reference_dir = os.path.join(base_path, parent_dir, reference_directory)
        reference_images_numbers = extract_image_numbers(os.listdir(reference_dir))

        max_image_number = max(reference_images_numbers)
        all_possible_images = set(str(i) + '.png' for i in range(1, max_image_number + 1))
        
        missing_files = all_possible_images - set(str(i) + '.png' for i in reference_images_numbers)

        # For each sub-directory, compare against 'image' directory
        for sub_dir_name in sub_directories:#[1:]:
            sub_dir = os.path.join(base_path, parent_dir, sub_dir_name)
            if os.path.exists(sub_dir):
                sub_files = set(os.listdir(sub_dir))
                missing_in_sub = all_possible_images - sub_files
                missing_in_sub = all_possible_images - sub_files
                missing_files = missing_files.union(missing_in_sub)

        # Update the blacklist
        blacklist[parent_dir] = natsort.natsorted(list(missing_files))

    # Save updated blacklist to file
    with open(blacklist_file, 'w') as f:
        json.dump(blacklist, f, indent=4)

    print(f"Updated blacklist saved to {blacklist_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--base_path', type=str,  help='.', default='/media/ssd1/hyunsoocha/GitHub/PointAvatar/data/datasets/total_composition_Dave/total_composition_Dave')
    parser.add_argument('--parent_directories', type=str, help='.', default='all')
    parser.add_argument('--sub_directories', type=str, help='.', default='mask mask_object')
    parser.add_argument('--reference_directory', type=str, help='.', default='image')
    args = parser.parse_args()

    if args.parent_directories == 'all':
        parent_directories = [d for d in os.listdir(args.base_path) if os.path.isdir(os.path.join(args.base_path, d))]
    else:
        parent_directories = list(filter(None, args.parent_directories.split()))
    sub_directories = list(filter(None, args.sub_directories.split()))
    base_path = args.base_path

    blacklist_maker(parent_directories, sub_directories, base_path, args.reference_directory)