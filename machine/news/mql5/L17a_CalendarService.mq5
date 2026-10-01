//+------------------------------------------------------------------+
//| L17a_CalendarService.mq5 - ML Bot. A SKETCH for prompt L17a.     |
//| Written and compiled in L12; NOT run in L12 (it reads 2026 news: |
//| only L17a may, under T14 "use: news-values = yes").              |
//|                                                                  |
//| An MT5 SERVICE (no chart needed; starts with the terminal once   |
//| added in Navigator > Services). Read only: no order, no price.   |
//|                                                                  |
//| Writes to the COMMON folder (%APPDATA%\MetaQuotes\Terminal\      |
//| Common\Files\ML-Bot\), which the bot can read without knowing    |
//| the terminal's own data folder:                                  |
//|  calendar_live.tsv   one line per NEW or CHANGED value (an       |
//|                      actual number coming out), appended, same   |
//|                      columns as L12_CalendarExport + the offset  |
//|  calendar_week.tsv   the next 7 days of releases, rewritten      |
//|                      whole every InpWeekMinutes (temp + move, so |
//|                      the bot never reads half a file)            |
//|  heartbeat.txt       last write time (GMT) + server offset: a    |
//|                      stale heartbeat = dead feed (likhith B21)   |
//|                                                                  |
//| Times are server time at the CURRENT offset (the L12 trap: MT5   |
//| converts with today's offset) - every line carries               |
//| server_minus_gmt_seconds, so utc = time - offset.                |
//+------------------------------------------------------------------+
#property service
#property copyright "ML Bot"
#property version   "1.00"

input string InpCountry     = "US";
input int    InpEverySec    = 5;
input int    InpWeekMinutes = 60;
input string InpFolder      = "ML-Bot";

string Cell(const long v) { return v == LONG_MIN ? "" : IntegerToString(v); }
string Stamp(const datetime t) { return TimeToString(t, TIME_DATE | TIME_SECONDS); }

string Line(const MqlCalendarValue &v, const long offset)
  {
   MqlCalendarEvent ev;
   if(!CalendarEventById(v.event_id, ev))
      return "";
   return StringFormat("%I64u\t%I64u\t%s\t%d\t%d\t%s\t%s\t%d\t%s\t%s\t%s\t%s\t%I64d",
                       v.id, v.event_id, ev.name, (int)ev.importance, (int)ev.time_mode, Stamp(v.time),
                       Stamp(v.period), v.revision, Cell(v.actual_value), Cell(v.forecast_value),
                       Cell(v.prev_value), Cell(v.revised_prev_value), offset);
  }

bool Append(const string name, const string text)
  {
   int h = FileOpen(name, FILE_READ | FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON | FILE_SHARE_READ, '\t', CP_UTF8);
   if(h == INVALID_HANDLE)
      return false;
   FileSeek(h, 0, SEEK_END);
   FileWriteString(h, text);
   FileClose(h);
   return true;
  }

void WriteWeek(const long offset)
  {
   MqlCalendarValue vals[];
   datetime now = TimeTradeServer();
   if(CalendarValueHistory(vals, now, now + 7 * 24 * 3600, InpCountry) <= 0)
      return;
   string tmp = InpFolder + "\\calendar_week.tmp";
   int h = FileOpen(tmp, FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON, '\t', CP_UTF8);
   if(h == INVALID_HANDLE)
      return;
   for(int i = 0; i < ArraySize(vals); i++)
     {
      string s = Line(vals[i], offset);
      if(s != "")
         FileWriteString(h, s + "\n");
     }
   FileClose(h);
   FileMove(tmp, FILE_COMMON, InpFolder + "\\calendar_week.tsv", FILE_COMMON | FILE_REWRITE);
  }

void OnStart()
  {
   ulong change_id = 0;
   MqlCalendarValue vals[];
   CalendarValueLast(change_id, vals, InpCountry);   // change_id 0 -> only learns the calendar's current state
   datetime lastWeek = 0;
   while(!IsStopped())
     {
      long offset = (long)(TimeTradeServer() - TimeGMT());
      if(CalendarValueLast(change_id, vals, InpCountry) > 0)
        {
         string text = "";
         for(int i = 0; i < ArraySize(vals); i++)
           {
            string s = Line(vals[i], offset);
            if(s != "")
               text += s + "\n";
           }
         Append(InpFolder + "\\calendar_live.tsv", text);
        }
      if(TimeGMT() - lastWeek >= InpWeekMinutes * 60)
        {
         WriteWeek(offset);
         lastWeek = TimeGMT();
        }
      int h = FileOpen(InpFolder + "\\heartbeat.txt", FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON, '\t', CP_UTF8);
      if(h != INVALID_HANDLE)
        {
         FileWriteString(h, "gmt=" + Stamp(TimeGMT()) + "\nserver_minus_gmt_seconds=" + IntegerToString(offset) + "\n");
         FileClose(h);
        }
      Sleep(InpEverySec * 1000);
     }
  }
//+------------------------------------------------------------------+
