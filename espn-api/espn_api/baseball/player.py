from .constant import POSITION_MAP, PRO_TEAM_MAP, STATS_MAP
from .utils import json_parsing
import pdb

class Player(object):
    '''Player are part of team'''
    def __init__(self, data, year):
        self.name = json_parsing(data, 'fullName')
        self.playerId = json_parsing(data, 'id')
        
        # Get eligible slots
        self.eligibleSlots = [POSITION_MAP.get(pos, str(pos)) for pos in json_parsing(data, 'eligibleSlots')]
        
        # Get all valid positions from eligible slots
        valid_positions = []
        for pos in self.eligibleSlots:
            # Handle composite positions
            if pos == '1B/3B':
                valid_positions.extend(['1B', '3B'])
            elif pos == '2B/SS':
                valid_positions.extend(['2B', 'SS'])
            elif pos == 'IF':
                valid_positions.extend(['1B', '2B', '3B', 'SS'])
            elif pos == 'OF':
                valid_positions.extend(['LF', 'CF', 'RF'])
            elif pos in ['C', '1B', '2B', '3B', 'SS', 'OF', 'LF', 'CF', 'RF', 'DH']:
                valid_positions.append(pos)
        
        # Remove duplicates and sort
        valid_positions = sorted(list(set(valid_positions)))
        
        # Set position to first valid position, or default position if none found
        default_pos = POSITION_MAP.get(json_parsing(data, 'defaultPositionId'), str(json_parsing(data, 'defaultPositionId')))
        self.position = valid_positions[0] if valid_positions else default_pos
        
        # Print debug info
        print(f"DEBUG - {self.name}")
        print(f"  Default Position: {default_pos}")
        print(f"  Valid Positions: {valid_positions}")
        print(f"  Selected Position: {self.position}")
        
        self.lineupSlot = POSITION_MAP.get(data.get('lineupSlotId'), '')
        self.acquisitionType = json_parsing(data, 'acquisitionType')
        self.proTeam = PRO_TEAM_MAP.get(json_parsing(data, 'proTeamId'), json_parsing(data, 'proTeamId'))
        self.injuryStatus = json_parsing(data, 'injuryStatus')
        self.stats = {}

        player = data.get('playerPoolEntry', {}).get('player') or data['player']
        self.injuryStatus = player.get('injuryStatus', self.injuryStatus)
        self.injured = player.get('injured', False)

        # add available stats
        player_stats = player.get('stats', [])
        for stats in player_stats:
            stats_split_type = stats.get('statSplitTypeId')
            if stats.get('seasonId') != year or (stats_split_type != 0 and stats_split_type != 5):
                continue
            stats_breakdown = stats.get('stats') or stats.get('appliedStats', {})
            breakdown = {STATS_MAP.get(int(k), k):v for (k,v) in stats_breakdown.items()}
            points = round(stats.get('appliedTotal', 0), 2)
            scoring_period = stats.get('scoringPeriodId')
            stat_source = stats.get('statSourceId')
            # TODO update stats to include stat split type (0: Season, 1: Last 7 Days, 2: Last 15 Days, 3: Last 30, 4: ??, 5: ?? Used in Box Scores)
            (points_type, breakdown_type) = ('points', 'breakdown') if stat_source == 0 else ('projected_points', 'projected_breakdown')
            if self.stats.get(scoring_period):
                self.stats[scoring_period][points_type] = points
                self.stats[scoring_period][breakdown_type] = breakdown
            else:
                self.stats[scoring_period] = {points_type: points, breakdown_type: breakdown}
        self.total_points = self.stats.get(0, {}).get('points', 0)
        self.projected_total_points = self.stats.get(0, {}).get('projected_points', 0)
            
    def __repr__(self):
        return 'Player(%s)' % (self.name, )
