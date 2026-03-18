redis-cli 指令
- config get save
- lastsave
- bgsave
- info memory
- INFO persistence
    - 重點看這幾個欄位：
    - rdb_last_save_time          → 上次成功時間
    - rdb_changes_since_last_save → 上次快照後有多少寫入
    - rdb_last_bgsave_status      → 上次快照是否成功
    - aof_enabled                 → AOF 是否開啟


看redis-stack寫入log
- docker logs redis-stack | grep -i "bgsave\|rdb\|save"