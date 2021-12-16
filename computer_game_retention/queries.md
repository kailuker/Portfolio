
```sql

SELECT joined AS day_joined, 
    COUNT(DISTINCT player_id) AS total_joined, -- counting how many players joined on a certain day
    SUM(CASE
        WHEN day_number >30 -- measuring whether the players have been retained
        THEN 1
        ELSE 0
        END) retention, -- finding the total number of players retained
    ROUND((SUM(CASE 
        WHEN day_number >30
        THEN 1
        ELSE 0
        END))/COUNT(DISTINCT player_id), 2) AS fractional_retention, -- dividing the number of players retained by the total number that joined that day
    ROUND(avg_wins_per_day, 2) AS avg_wins_per_day -- finding the average win rate grouped by join day
FROM 
    (SELECT -- getting all needed values from both tables
        player.player_id,
        player.joined,
        matches.last_match,
        matches.last_match - player.joined AS day_number,
        AVG(wins) OVER(PARTITION BY player.joined) AS avg_wins_per_day
        FROM 
            (SELECT 
                pla.player_id,
                pla.joined,
            FROM `graphic-tensor-329514.sql_project.player_info` AS pla
            GROUP BY pla.player_id, pla.joined) AS player -- getting values from player_info table
        FULL JOIN
            (SELECT 
                m.player_id,
                MAX(m.day) AS last_match,
                SUM(CASE WHEN m.outcome = 'win' 
                    THEN 1 
                    ELSE 0 
                    END) AS wins
            FROM `graphic-tensor-329514.sql_project.matches_info` AS m
            GROUP BY m.player_id) AS matches -- getting values from matches_info table
        ON player.player_id = matches.player_id) -- joining the matches_info & player_info tables
GROUP BY day_joined, avg_wins_per_day
ORDER BY day_joined
LIMIT 334 -- players who joined less than 30 days before the end of the year will appear to not be retained, so exclude those rows

```

