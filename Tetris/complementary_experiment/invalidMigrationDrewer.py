import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sxy_invalidMigrationNum = [1047,1154,1241,1286,1330,1356]
invalidMigrationNum=[7940,16488,25364,34461,43654,52924]
windowSize=[1,2,3,4,5,6]

fig, ax1 = plt.subplots(figsize=(12, 8))

ax1.set_xlabel('Size of $\mathit{w}$', fontsize = 40)
ax1.set_ylabel('#invalid migrations', color = '#2F7FC1', fontsize = 40)
ax1.tick_params(width = 4, length = 8, labelsize = 36)
ax1.tick_params(axis='y', which='major', width=4, length=8, labelsize=36, colors='#2F7FC1')
bwith = 4
ax1.spines['top'].set_linewidth(bwith)
ax1.spines['bottom'].set_linewidth(bwith)
ax1.spines['left'].set_linewidth(bwith)
ax1.spines['right'].set_linewidth(bwith)

plt.plot(windowSize, sxy_invalidMigrationNum, color = '#2F7FC1', linewidth = 6.0, marker = 'o', ms = 15, label='v2018')
plt.grid(axis = 'y')

# 创建第二个 y 轴
ax2 = ax1.twinx()
ax2.set_ylabel('#invalid migrations', color='red', fontsize=40)
ax2.tick_params(width=4, length=8, labelsize=36)
ax2.tick_params(axis='y', which='major', width=4, length=8, labelsize=36, colors='red')
ax2.spines['top'].set_linewidth(4)
ax2.spines['bottom'].set_linewidth(4)
ax2.spines['left'].set_linewidth(4)
ax2.spines['right'].set_linewidth(4)

ax2.plot(windowSize, invalidMigrationNum, color='red', linewidth=4.0, marker='s', ms=15, label='v2022')

# 自定义图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines = lines1 + lines2
labels = labels1 + labels2
ax1.legend(lines, labels, loc='best', fontsize=40)

plt.savefig('./invaildMigrationNum.pdf', bbox_inches = 'tight', transparent = True)
plt.close()