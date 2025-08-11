import re
import pandas as pd
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
