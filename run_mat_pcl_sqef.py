import os
import pickle
import numpy as np

from dual_contouring.sdf_mat_field_extract_mesh import extract_octree_by_sqef
from dual_contouring.neural_utils import *
import numpy as np
from config import get_sqef_extraction_args
import time
def main(args):
    npz_path=os.path.join(args.input_dir, args.dataset+".sampling.npz")
    data=np.load(npz_path,allow_pickle=True)
    start_time = time.time()
    octree_data,octree=extract_octree_by_sqef(data,args,sampling_depth=args.sampling_depth)
    end_time = time.time()
    run_time = end_time - start_time
    os.makedirs(args.output_dir, exist_ok=True)
    if args.debug:
        savepath=os.path.join(args.output_dir,args.dataset+".depth"+str(args.max_depth)+".sqef.npz")
        np.savez(savepath,**octree_data)
        print(f"generate {savepath} finished!")
        # savepath=os.path.join(args.output_dir,args.dataset+".depth"+str(args.max_depth)+".sqef.pkl")
        # with open(savepath, 'wb') as f:
        #     pickle.dump(octree, f)
        #print(f"generate {savepath} finished!")
        
        # savepath=os.path.join(args.output_dir,args.dataset+".args.txt")
        # with open(savepath, 'w') as f:
        #     f.write(f'max_depth:{args.max_depth}\n')
        #     f.write(f'sampling_depth:{args.sampling_depth}\n')
        #     f.write(f'margin_threshold:{args.margin_threshold}\n')
        #     f.write(f'singular_value_threshold:{args.singular_value_threshold}\n')
        # print(f"generate {savepath} finished!")
    savepath=os.path.join(args.output_dir,args.dataset+".depth"+str(args.max_depth)+".sqef.ma")
    octree.save_ma_mesh(savepath)
    print(f"generate {savepath} finished!")
    savepath=os.path.join(args.output_dir,args.dataset+".depth"+str(args.max_depth)+".tri.sqef.obj")
    octree.save_medial_mesh(savepath)
    print(f"generate {savepath} finished!")    
    print(f"cost time:{run_time:.6f} sec!")

if __name__ == "__main__":
    main(get_sqef_extraction_args())

