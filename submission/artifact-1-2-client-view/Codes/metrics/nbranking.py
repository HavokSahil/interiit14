from typing import List
from db.bmrep_db import BeaconMeasurementDB
from db.nbrank_db import NeighborRankingDB
from db.neighbor_db import NeighborDB
from db.station_db import StationDB
from logger import Logger
from model.neighbor import Neighbor
from model.measurement import BeaconMeasurement
from model.station import Station
from parser.neighbor_parser import neighbor_from_beacon_report

class NeighborRanking:
    def __init__(self, stdb: StationDB, nrdb: NeighborRankingDB, bmr: BeaconMeasurementDB, ndb: NeighborDB):
        self.nrdb = nrdb
        self.bmr = bmr
        self.ndb = ndb
        self.stdb = stdb

    @staticmethod
    def _rank_beacon(station: Station, lbmr: List[BeaconMeasurement], nb: NeighborDB) -> List[Neighbor]:
        """
        Create Neighbor objects from BeaconReport entries and rank them.
        Scoring:
          score = rssi_dbm + 0.5 * snr_db + (known_neighbor ? bonus : 0)
        Higher score == better neighbor.
        """
        scored: List[tuple[float, Neighbor]] = []

        for bmr in lbmr:
            for report in bmr.beacon_reports:
                try:
                    # build neighbor from beacon report
                    n = neighbor_from_beacon_report(report)

                    # compute basic score
                    # rssi_dbm is usually negative (e.g. -50). Use it directly (less negative = better)
                    rssi = report.rssi_dbm
                    snr = report.snr_db
                    score = rssi + 0.5 * snr

                    # small bonus if this neighbor is present in NeighborDB for this station
                    if nb.get(report.bssid, station.mac) is not None:
                        score += 5.0

                    scored.append((score, n))
                except Exception as e:
                    # skip malformed reports but continue
                    Logger.log_info(f"_rank_beacon: {e}")
                    continue

        # sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)
        return [n for _, n in scored]

    @staticmethod
    def _rank_nobeacon(station: Station, nb: NeighborDB) -> List[Neighbor]:
        """
        Fallback ranking when no beacon measurements are available.
        Simply returns the NeighborDB entries for that station ordered:
          1) neighbors with an SSID first (likely real APs)
          2) then by BSSID lexicographically
        """
        neighbors = nb.all_for_sta(station.mac)
        # stable sort: first by BSSID, then bring SSID-bearing entries first
        neighbors_sorted = sorted(
            neighbors,
            key=lambda n: (0 if (getattr(n, "ssid", None)) else 1, n.bssid or "")
        )
        # return a shallow copy to avoid caller mutating DB objects
        return [n for n in neighbors_sorted]

    def update(self):
        for station in self.stdb.all():
            mac = station.mac
            lbmr = self.bmr.get(mac)
            ranking = []
            if lbmr:
                ranking = NeighborRanking._rank_beacon(station, lbmr, self.ndb)
            else:
                ranking = NeighborRanking._rank_nobeacon(station, self.ndb)
            self.nrdb.set_ranking(mac, ranking)
