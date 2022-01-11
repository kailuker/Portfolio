```sql
SELECT joined AS day_joined,
       COUNT(DISTINCT player.player_id) AS total_joined, -- counting how many players joined on a certain day
       SUM(CASE WHEN matches.last_match - player.joined >30 -- measuring whether they played 30 or more days after joining
           THEN 1
           ELSE 0
           END) AS retention, -- finding the total number of players retained
       ROUND((SUM(CASE WHEN matches.last_match - player.joined >30
                   THEN 1
                   ELSE 0
                   END))/COUNT(DISTINCT player.player_id), 4) AS fractional_retention, -- dividing the number of players retained by the total number that joined that day
        ROUND((SUM(wins)/SUM(num_matches)), 4) AS win_rate -- dividing the total number of wins by players that joined that day by total number of matches played by players who joined that day
FROM 
        (SELECT player_id,
                joined
         FROM `graphic-tensor-329514.sql_project.player_info`) AS player
    JOIN
        (SELECT player_id,
                MAX(day) AS last_match, -- finding the day of each players' most recent match
                SUM(CASE WHEN outcome = 'win' 
                    THEN 1 
                    ELSE 0 
                    END) AS wins, -- counting the number of times each player wins a match
                COUNT(DISTINCT match_id) AS num_matches -- counting the total number of matches each player played
         FROM `graphic-tensor-329514.sql_project.matches_info`
         GROUP BY player_id) AS matches  -- aggregating values by player_id
    ON player.player_id = matches.player_id -- joining player_info & matches_info tables
GROUP BY day_joined -- aggregating results by players' join day
ORDER BY day_joined
LIMIT 334 -- players who joined less than 30 days before the end of the year will appear to not be retained, so exclude those rows
```