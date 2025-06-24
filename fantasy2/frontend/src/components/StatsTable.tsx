// Utility to safely render any value
function safeRender(val: any) {
  if (typeof val === 'string' || typeof val === 'number') return val;
  if (val === null || val === undefined) return '';
  return JSON.stringify(val);
}

const StatsTable: React.FC<StatsTableProps> = ({ data, playerType, statType, selectedTeams }) => {
                <TableCell sx={teamColumnStyle} className="team-ellipsis-cell">
                  <span className="team-ellipsis-span">
                    {safeRender(player.team)}
                  </span>
                </TableCell> 
} 