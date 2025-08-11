import re
import pandas as pd
<<<<<<< HEAD
def convert_greybody_file(file_path, output_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    case_splits = re.split(r"Case number:\s+\d+", content)[1:]

    def parse_case_block_fixed(block):
        import re
        lines = block.strip().splitlines()
        sp = float(re.search(r"Surface pressure \(Pa\):\s+([0-9.]+)", block).group(1))
        gi = next(i for i, line in enumerate(lines) if 'solar_cosw_atten' in line)
        gvals = list(map(float, lines[gi + 1].strip().split()))
        solar_cosw_atten, window, tau0, tau_total = gvals
        data = []
        for line in lines[gi + 3:]:
            try:
                parts = list(map(float, line.strip().split()))
                if len(parts) == 5:
                    data.append(parts)
            except ValueError:
                break
        import pandas as pd
        df = pd.DataFrame(data, columns=["p", "q", "t", "rad_up", "rad_down"])
        df = df.drop(columns=["q"])
        for name, val in zip(["surface_pressure", "solar_cosw_atten", "window", "tau0", "tau_total"],
                             [sp, solar_cosw_atten, window, tau0, tau_total]):
            df[name] = val
        return df

    import pandas as pd
    all_cases = [parse_case_block_fixed(block) for block in case_splits]
    df_all = pd.concat(all_cases, ignore_index=True)
    df_all.to_csv(output_path, index=False)

convert_greybody_file("data.txt", "atmospheric_radiation_dataset.csv")
=======

# 理想气体常数 (单位：J/(kg·K))
R_air = 287.05

# 读取原始数据
file_path = 'data.txt'  # 请根据实际情况修改文件路径
with open(file_path, 'r') as f:
    raw_text = f.read()

# 分割为不同的Case
case_pattern = re.compile(r'Case number:\s+\d+\s+Surface pressure.*?(?=(?:Case number:|\Z))', re.S)
cases = case_pattern.findall(raw_text)

structured_data = []

for case in cases:
    # 提取Case元数据
    header_match = re.search(r'Case number:\s+(\d+)\s+Surface pressure \(Pa\):\s+([\d\.E\+\-]+)', case)
    if not header_match:
        continue
    case_num = int(header_match.group(1))
    surface_pressure = float(header_match.group(2))
    
    # 提取每层数据
    layer_pattern = re.compile(r'^\s*([\d\.E\+\-]+)\s+[\d\.E\+\-]+\s+([\d\.E\+\-]+)\s+([\d\.E\+\-]+)\s+([\d\.E\+\-]+)', re.M)
    layers = layer_pattern.findall(case)
    
    for layer in layers:
        p_half = float(layer[0])  # 气压p
        temp = float(layer[1])  # 温度T
        up_flux = float(layer[2])  # 向上辐射通量
        down_flux = float(layer[3])  # 向下辐射通量
        
        # 计算密度 (单位：kg/m³)
        density = p_half / (R_air * temp)
        
        structured_data.append({
            'case': case_num,
            'surface_pressure': surface_pressure,
            'p': p_half,
            'T': temp,
            'density': density,
            'up_flux': up_flux,
            'down_flux': down_flux
        })

# 创建DataFrame
df = pd.DataFrame(structured_data)

# 保存为CSV文件
output_path = 'atmospheric_radiation_dataset.csv'
df.to_csv(output_path, index=False)

print(f"CSV文件已生成并保存为: {output_path}")

>>>>>>> 7e396bc (revised radiation_nn.py)
