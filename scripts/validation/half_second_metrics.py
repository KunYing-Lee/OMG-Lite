import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('output', type=Path)
p=parser.parse_args().output
r={}
for name in ['walk','squat','hands','turn']:
 r[name]={}
 for label,period in [('half',25),('second',50)]:
  d=p/f'{label}_{name}';x=np.load(d/'rollout.npz');q=x['reference'];e=x['executed'];seams=np.arange(period,len(q)-1,period)
  rootv=np.diff(q[:,:3],axis=0)*50;jointv=np.diff(q[:,7:],axis=0)*50
  yaw=lambda a:np.unwrap(Rotation.from_quat(a[:,[4,5,6,3]]).as_euler('xyz')[:,2])
  ry=yaw(q);ey=yaw(e)
  events=json.loads((d/'report.json').read_text())['events']
  assert [v['request_frame'] for v in events]==list(range(0 if period==25 else 25,500,period))
  assert all(v['preview_deadline_frame']-v['request_frame']==14 for v in events)
  r[name][label]={
   'frames':len(q),'period_frames':period,'replans':len(events),
   'root_rmse_cm':float(np.sqrt(np.mean(np.sum((e[:,:3]-q[:,:3])**2,axis=1)))*100),
   'joint_rmse_rad':float(np.sqrt(np.mean((e[:,7:]-q[:,7:])**2))),
   'root_seam_step_cm_max':float(np.max(np.linalg.norm(q[seams,:3]-q[seams-1,:3],axis=1))*100),
   'joint_seam_step_rad_max':float(np.max(abs(q[seams,7:]-q[seams-1,7:]))),
   'root_seam_velocity_change_m_s_p95':float(np.percentile(np.linalg.norm(rootv[seams-1]-rootv[seams-2],axis=1),95)),
   'joint_seam_velocity_change_rad_s_p95':float(np.percentile(np.sqrt(np.mean((jointv[seams-1]-jointv[seams-2])**2,axis=1)),95)),
   'reference_yaw_change_deg':float(np.rad2deg(ry[-1]-ry[0])),
   'execution_yaw_change_deg':float(np.rad2deg(ey[-1]-ey[0])),
   'reference_root_z_min':float(q[:,2].min()),'execution_root_z_min':float(e[:,2].min()),
   'reference_distance_xy_m':float(np.linalg.norm(q[-1,:2]-q[0,:2])),
   'execution_distance_xy_m':float(np.linalg.norm(e[-1,:2]-e[0,:2])),
  }
 first=[np.load(p/f'{label}_{name}'/'plan_000.npz')['qpos'] for label in ['half','second']]
 r[name]['initial_plan_max_abs_diff']=float(np.max(abs(first[0]-first[1])))
(p/'metrics.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
