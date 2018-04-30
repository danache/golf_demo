import configparser
import os
def process_network(conf_file):
    params = {}
    config = configparser.ConfigParser()
    config.read(conf_file)
    for section in config.sections():
        if section == 'Network':
            for option in config.options(section):
                params[option] = eval(config.get(section, option))
    return params

def generate_sys_Log(args):
    video_path = args.video_path
    output_path = args.output_path
    use_mask_RCNN = args.detection
    human_height = args.height
    model_type = args.model_type
    str = "Video_path is : {}".format(video_path)
    str += "\n" +"Output_path is : {}".format(output_path)
    if use_mask_RCNN:
        str += "\n" + " Use Mask-RCNN detection results"
    else:
        str += "\n" + " Use SSD detection results"
    str += "\n" + "human_height is : {}".format(human_height)
    if model_type == 0:
        str += "\n" + " Use Hourglass results"
    elif model_type == 1:
        str += "\n" + " Use CPN results"
    elif model_type == 2:
        str += "\n" + " Use mix results"
    else:
        print("Wrong model type !")
        exit(-1)
    print(str)
    return video_path, output_path, use_mask_RCNN, human_height, model_type


def check_dir(file_path):
    if os.path.isdir(file_path):
        return
    os.mkdir(file_path)
    return


def videos2frames(file_path, out_dir):
    check_dir(out_dir)
    return_lst = []
    if os.path.isdir(file_path):
        for root, dirs, files in os.walk(file_path, topdown=False):
            for name in files:
                if name[-3:] == 'MOV' or name[-3:] == 'mp4':
                    outpath = os.path.join(out_dir, name[:-4])
                    check_dir(outpath)
                    video_name = os.path.join(root, name)
                    prename = name[:-4]

                    os.system('ffmpeg  -i %s -vsync 2  -f image2 %s/%s-frame%%05d.png' % (video_name, outpath, prename))
                    print("{} is done ! frame dir : {}".format(video_name, outpath))
                    return_lst.append(outpath)
    return return_lst


