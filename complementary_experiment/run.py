from sandpiper import Sandpiper_algo1
from dataTransform import getMappingAndUtil_pro, getMappingUtilForPlacement
from utl import recordInitTrace, recordTrace_pro,getInvalidMigrationNum, getNumOfLongTravel, getTotalMigrationNum, getMapping_Pro
import os
import json
import logging

logging.basicConfig(level=logging.DEBUG)

def get_dataInTrace(day, cur_hour):
    return getMappingAndUtil_pro(os.path.join(datapath, f'MSMetrics_{day}d.csv'), os.path.join(datapath, f'NodeMetrics_{day}d.csv'), cur_hour)

w=6
hour=0
longTravelNum=0
inMigNum=0
datapath='./data/processed_data'

while(hour<13*24):
    cur_hour=hour
    day=int(cur_hour / 24) +1

    i=0
    logging.info(f'从{hour}h 开始，窗口大小为{w}')

    try:
        [sorted_df, cpu_t0, mem_t0, x_t0, flag]=get_dataInTrace(day,cur_hour)
        while flag==0:
            cur_hour+=1
            day=int(cur_hour / 24) +1
            i+=1
            [sorted_df, cpu_t0, mem_t0, x_t0, flag]=get_dataInTrace(day,cur_hour)
            
        # 记录每个容器的轨迹
        trace=recordInitTrace(sorted_df)

        while i<w+1:
            logging.info(f'i=={i}，cur_hour={cur_hour}')
            placement=Sandpiper_algo1(x_t0, cpu_t0, mem_t0, 75, 75)
            # logging.info(f'---第{i+1}次迁移完成----')
            mapping=getMapping_Pro(placement, sorted_df) # placement转mapping, placement中可能存在key-['None']的项
            # 再次记录每个容器的轨迹
            trace=recordTrace_pro(trace, mapping)
            # logging.info(f'---第{i+1}次记录完成----')
            if i==w:
                break
            # 得到下一时刻的各种数据
            cur_hour+=1
            day=int(cur_hour / 24) +1
            [sorted_df, cpu_t0, mem_t0, x_t0, flag]=get_dataInTrace(day,cur_hour)
            while flag==0 and i<w: # 这个时刻trace没有数据，不记录，跳过，直到获取到数据。退出while：1、获取到数据了 或者 2、i==w到窗口上限了
                logging.info(f'第{cur_hour}h 的数据缺失，跳过。。。')
                cur_hour+=1
                day=int(cur_hour / 24) +1
                i+=1
                [sorted_df, cpu_t0, mem_t0, x_t0, flag]=get_dataInTrace(day,cur_hour)
            if i==w:
                break
            mapping=getMapping_Pro(x_t0, sorted_df)
            trace=recordTrace_pro(trace, mapping)
            i+=1

        logging.info(f'从{hour}h 开始的迁移完成')
        trace_json=json.dumps(trace, indent=0)
        with open(f'trace_{hour}h.txt', 'w') as f:
            f.write(trace_json)
            f.write('\n')
        longTravelNum+=getNumOfLongTravel(trace)
        # tolMigNum+=getTotalMigrationNum(trace)
        inMigNum+=getInvalidMigrationNum(trace)
        logging.info(f'目前无效迁移总数为{inMigNum}, 长迁移总数为{longTravelNum}')
    except Exception as e:
        logging.error(f'处理 {hour}h 发生异常: {str(e)}')
    finally:
        hour+=1

logging.info(f'总无效迁移数: {inMigNum}, 总长距离无效迁移数: {longTravelNum}')














