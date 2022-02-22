import decimal
import pandas as pd
from decimal import Decimal, getcontext
from operator import attrgetter
from core.service import ServiceProfile

# https://minorman.tistory.com/118
getcontext().rounding = decimal.ROUND_DOWN


class CSVReader:
    def __init__(self, filename, num_edgeDCs):
        self.filename = filename
        df = pd.read_csv(self.filename)

        self.service_profiles = []
        for i in range(len(df)):
            row = df.iloc[i]
            # !note: +1 to avoid edgeDCs[0] as the central DC
            user_loc = (row.user_loc.astype(dtype=int) % num_edgeDCs) + 1
            self.service_profiles.append(
                ServiceProfile(row.service_id.astype(dtype=int),
                               row.submit_time.astype(dtype=int),
                               row.plan_cpu.astype(dtype=int),
                               round(Decimal(row.plan_mem), 9),
                               round(Decimal(row.plan_disk), 9),
                               row.duration,
                               user_loc,
                               row.e2e_latency.astype(dtype=int)))

        self.service_profiles.sort(key=attrgetter('submit_time'))

    def generate(self, offset, number):
        number = number if offset + number < len(self.service_profiles) else len(self.service_profiles) - offset
        return self.service_profiles[offset:offset+number]

