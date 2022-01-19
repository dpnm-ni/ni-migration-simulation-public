import decimal
import pandas as pd
import numpy as np
from decimal import Decimal, getcontext
from operator import attrgetter
from core.service import ServiceProfile


getcontext().rounding = decimal.ROUND_DOWN


class CSVReader:
    def __init__(self, filename):
        self.filename = filename
        df = pd.read_csv(self.filename)

        self.service_profiles = []
        for i in range(len(df)):
            row = df.iloc[i]

            # FIXME: final alg performance become different due to this randomness
            # identify user location (edge machine) of the service
            edge_machine_id = row.edge_id.astype(dtype=str)
            if len(edge_machine_id) == 4:
                edge_machine_id = np.random.randint(0, 13)
            else:
                edge_machine_id = int(edge_machine_id[0])

            self.service_profiles.append(
                ServiceProfile(row.service_id.astype(dtype=int),
                               row.submit_time.astype(dtype=int),
                               row.plan_cpu.astype(dtype=int),
                               # https://minorman.tistory.com/118
                               round(Decimal(row.plan_mem), 9),
                               round(Decimal(row.plan_disk), 9),
                               row.duration,
                               edge_machine_id))

        self.service_profiles.sort(key=attrgetter('submit_time'))

    def generate(self, offset, number):
        number = number if offset + number < len(self.service_profiles) else len(self.service_profiles) - offset
        return self.service_profiles[offset:offset+number]
